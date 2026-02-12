"""
Hybrid Supplier Search Service

Combines internal database vector search with real-time web discovery
using Amazon Nova web grounding to find suppliers for projects.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db_context
from app.core.logging_config import get_logger
from app.models.database import Project, ProjectSupplier, Supplier
from app.services.bedrock_converse_service import BedrockConverseService
from app.services.embedding_service import get_embedding_service
from app.services.supplier_deduplication import (
    get_existing_suppliers_for_project,
    normalize_supplier_name,
    check_supplier_exists,
)

logger = get_logger(__name__)


class HybridSupplierSearchService:
    """
    Service for hybrid supplier discovery combining:
    1. Internal database vector similarity search
    2. Amazon Nova web grounding for real-time web discovery

    Internal suppliers are prioritized (boosted) over web suppliers as they
    are verified and already in our system.
    """

    # Boost factor for internal (verified) suppliers to prioritize them over web results
    # This is added to the similarity score when ranking combined results
    INTERNAL_SUPPLIER_BOOST = 0.10

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.converse_service = BedrockConverseService()

    async def search_internal(
        self,
        query: str,
        l1_category: str | None = None,
        l2_category: str | None = None,
        l3_category: str | None = None,
        region: str | None = None,
        limit: int = 50,
        threshold: float = 0.2,  # Minimum 20% vector similarity for quality results
    ) -> list[dict[str, Any]]:
        """
        Search for suppliers in the internal database using vector similarity.

        Args:
            query: Search query describing supplier requirements
            l1_category: Optional L1 category filter (matches capabilities->>'l1_category')
            l2_category: Optional L2 category filter (matches capabilities->>'l2_category')
            l3_category: Optional L3 category filter (matches capabilities->>'l3_category')
            region: Optional region filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of matching suppliers with similarity scores
        """
        try:
            query_embedding = await self.embedding_service.embed(query)

            async with get_db_context() as session:
                # Build dynamic WHERE clauses
                # Note: Use CAST(:param AS vector) instead of :param::vector
                # because asyncpg gets confused by :: right after a named parameter
                where_clauses = [
                    "is_active = true",
                    "embedding IS NOT NULL",
                    "1 - (embedding <=> CAST(:embedding AS vector)) > :threshold"
                ]
                params: dict[str, Any] = {
                    "embedding": str(query_embedding),
                    "threshold": threshold,
                    "limit": limit,
                }

                # Category filtering is OPTIONAL - we rely primarily on vector similarity
                # The LLM-generated semantic query should already capture the intent
                # Category filters can be added later for boosting/ranking if needed
                #
                # Note: If your suppliers have l1_category/l2_category/l3_category in capabilities,
                # you could uncomment the below to add filtering. For now, we skip it to ensure
                # we always get results from vector similarity.
                #
                # category_filters = []
                # if l1_category:
                #     category_filters.append("capabilities->>'l1_category' ILIKE :l1_category")
                #     params["l1_category"] = f"%{l1_category}%"
                # if category_filters:
                #     where_clauses.append(f"({' OR '.join(category_filters)})")

                where_clause = " AND ".join(where_clauses)

                sql = text(f"""
                    SELECT
                        id, name, description, categories, capabilities,
                        contact_info, performance_score, compliance_status,
                        metadata,
                        1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                    FROM suppliers
                    WHERE {where_clause}
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :limit
                """)

                result = await session.execute(sql, params)

                suppliers = []
                for row in result.fetchall():
                    similarity = round(float(row.similarity), 4)
                    # Apply boost to internal suppliers for ranking purposes
                    boosted_score = min(1.0, similarity + self.INTERNAL_SUPPLIER_BOOST)

                    supplier_data = {
                        "id": str(row.id),
                        "name": row.name,
                        "description": row.description,
                        "categories": row.categories or [],
                        "capabilities": row.capabilities or {},
                        "contact_info": row.contact_info or {},
                        "performance_score": float(row.performance_score) if row.performance_score else None,
                        "compliance_status": row.compliance_status,
                        "metadata": row.metadata or {},
                        "similarity": similarity,
                        "boosted_score": boosted_score,  # Score used for ranking (includes verification boost)
                        "source": "internal",
                        "verified": True,  # Internal suppliers are pre-verified
                    }

                    # Calculate tier based on boosted score
                    if boosted_score >= 0.85:
                        supplier_data["tier"] = "A"
                    elif boosted_score >= 0.7:
                        supplier_data["tier"] = "B"
                    else:
                        supplier_data["tier"] = "C"

                    suppliers.append(supplier_data)

                logger.info(
                    "Internal supplier search completed",
                    extra={
                        "query": query[:50],
                        "l1_category": l1_category,
                        "l2_category": l2_category,
                        "l3_category": l3_category,
                        "results_count": len(suppliers)
                    }
                )

                return {"suppliers": suppliers, "error": None, "status": "success"}

        except Exception as e:
            logger.error(f"Internal supplier search error: {e}", exc_info=True)
            return {"suppliers": [], "error": str(e), "status": "error"}

    async def search_web(
        self,
        query: str,
        category: str | None = None,
        region: str | None = None,
        limit: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Search for suppliers using Amazon Nova web grounding.

        Args:
            query: Search query describing supplier requirements
            category: Optional category to focus the search
            region: Optional region to focus the search
            limit: Maximum number of results

        Returns:
            List of web-discovered suppliers with source citations
        """
        try:
            # Build search prompt for web grounding
            search_context = []
            if category:
                search_context.append(f"Category: {category}")
            if region:
                search_context.append(f"Region: {region}")

            context_str = "\n".join(search_context) if search_context else ""

            system_prompt = f"""You are a supplier research assistant. Find real suppliers for the given requirements.

{context_str}

For each supplier found, extract:
1. Company name (official name)
2. Brief description of services
3. Website URL (official website only)
4. Location/headquarters
5. Key capabilities relevant to the query
6. Any certifications or compliance information
7. Employee count (approximate number of employees, e.g., 5000, 50000)
8. Annual revenue (approximate, e.g., "$50M", "$1B", "$500M")
9. Geographic coverage/regions served (e.g., ["North America", "Europe", "Asia Pacific"])
10. Main products and services offered

Return the results as a JSON array with this structure:
[
  {{
    "name": "Company Name",
    "description": "Brief description of services",
    "website": "https://official-website.com",
    "location": "City, Country",
    "capabilities": ["capability1", "capability2"],
    "certifications": ["ISO 9001", "SOC 2"],
    "employee_count": 5000,
    "revenue": "$500M",
    "coverage": ["North America", "Europe"],
    "products_and_services": ["Product 1", "Service 1", "Service 2"]
  }}
]

Find up to {limit} relevant suppliers. Only include real companies with verifiable information."""

            user_message = f"Find suppliers for: {query}"

            messages = [{"role": "user", "content": [{"text": user_message}]}]

            response = await self.converse_service.converse(
                messages=messages,
                system=system_prompt,
                enable_grounding=True,
                max_tokens=2048,
                temperature=0.3,  # Lower temperature for factual responses
            )

            response_text = response.get("text", "")
            citations = response.get("citations", [])

            # Parse supplier data from response
            suppliers, parse_error = self._parse_web_suppliers(response_text, citations)

            logger.info(
                "Web supplier search completed",
                extra={
                    "query": query[:50],
                    "category": category,
                    "region": region,
                    "results_count": len(suppliers),
                    "citations_count": len(citations)
                }
            )

            return {"suppliers": suppliers[:limit], "error": parse_error, "status": "success" if not parse_error else "partial"}

        except Exception as e:
            logger.error(f"Web supplier search error: {e}", exc_info=True)
            return {"suppliers": [], "error": str(e), "status": "error"}

    # Blocklist patterns for invalid supplier names (LLM instruction leakage)
    INVALID_NAME_PATTERNS = [
        "format your response",
        "here is",
        "here are",
        "the following",
        "please note",
        "note:",
        "example",
        "unknown",
        "n/a",
        "not available",
        "company name",
        "supplier name",
        "insert",
        "placeholder",
        "[",
        "]",
        "{",
        "}",
    ]

    def _is_valid_supplier_name(self, name: str) -> bool:
        """
        Validate that a supplier name is legitimate and not LLM instruction leakage.

        Args:
            name: Supplier name to validate

        Returns:
            True if valid, False if invalid
        """
        if not name or not isinstance(name, str):
            return False

        name_lower = name.lower().strip()

        # Must be at least 2 characters
        if len(name_lower) < 2:
            return False

        # Must not be longer than 200 characters (reasonable company name limit)
        if len(name_lower) > 200:
            return False

        # Check against blocklist patterns
        for pattern in self.INVALID_NAME_PATTERNS:
            if pattern in name_lower:
                logger.warning(
                    f"Rejected invalid supplier name (matched blocklist pattern '{pattern}')",
                    extra={"supplier_name": name}
                )
                return False

        # Must contain at least one letter (not just numbers/punctuation)
        if not any(c.isalpha() for c in name):
            return False

        # Reject if it looks like JSON or markdown formatting
        if name_lower.startswith(("```", "---", "##", "**", "__")):
            logger.warning(
                "Rejected invalid supplier name (markdown formatting)",
                extra={"supplier_name": name}
            )
            return False

        return True

    def _parse_web_suppliers(
        self,
        response_text: str,
        citations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Parse supplier data from the web grounding response.

        Args:
            response_text: LLM response text
            citations: Citations from web grounding

        Returns:
            Tuple of (list of parsed supplier dictionaries, parse error string or None)
        """
        suppliers = []
        parse_error: str | None = None

        # Try to extract JSON array from response using multiple strategies
        json_text: str | None = None

        # Strategy 1: Extract from markdown code block (```json ... ```)
        if "```" in response_text:
            code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if code_match:
                candidate = code_match.group(1).strip()
                if candidate.startswith("["):
                    json_text = candidate

        # Strategy 2: Find JSON array-of-objects pattern [{ ... }]
        # Uses bracket counting instead of greedy regex to avoid capturing
        # stray brackets in surrounding prose (e.g. "suppliers [based on search]:")
        if not json_text:
            arr_start = re.search(r'\[\s*\{', response_text)
            if arr_start:
                start = arr_start.start()
                depth = 0
                for i in range(start, len(response_text)):
                    if response_text[i] == '[':
                        depth += 1
                    elif response_text[i] == ']':
                        depth -= 1
                        if depth == 0:
                            json_text = response_text[start:i + 1]
                            break

        # Strategy 3: Entire response might be a raw JSON array
        if not json_text:
            stripped = response_text.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                json_text = stripped

        if json_text:
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("name"):
                            supplier_name = item.get("name", "").strip()

                            # Validate supplier name to prevent LLM instruction leakage
                            if not self._is_valid_supplier_name(supplier_name):
                                continue

                            supplier = {
                                "name": supplier_name,
                                "description": item.get("description", ""),
                                "website": item.get("website", ""),
                                "location": item.get("location", ""),
                                "capabilities": item.get("capabilities", []),
                                "certifications": item.get("certifications", []),
                                "employee_count": item.get("employee_count"),
                                "revenue": item.get("revenue"),
                                "coverage": item.get("coverage", []),
                                "products_and_services": item.get("products_and_services") or item.get("capabilities", []),
                                "source": "web_discovery",
                                "source_urls": [item.get("website")] if item.get("website") else [],
                            }
                            suppliers.append(supplier)
            except json.JSONDecodeError as exc:
                parse_error = "Web search response could not be parsed"
                logger.warning(
                    "Failed to parse JSON from web grounding response",
                    extra={"json_excerpt": json_text[:300], "error": str(exc)},
                )
        else:
            parse_error = "Web search returned no structured supplier data"
            logger.warning(
                "No JSON array found in web grounding response",
                extra={"response_excerpt": response_text[:300]},
            )

        # Add citation URLs to suppliers
        if citations:
            citation_urls = [c.get("url") for c in citations if c.get("url")]
            for supplier in suppliers:
                # Match citations to suppliers by website domain
                supplier_domain = self._extract_domain(supplier.get("website", ""))
                matching_urls = [
                    url for url in citation_urls
                    if supplier_domain and supplier_domain in url
                ]
                if matching_urls:
                    supplier["source_urls"] = list(set(supplier.get("source_urls", []) + matching_urls))

        return suppliers, parse_error

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for matching."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    async def hybrid_search(
        self,
        project_id: uuid.UUID,
        query: str | None = None,
        l1_category: str | None = None,
        l2_category: str | None = None,
        region: str | None = None,
        internal_limit: int = 50,
        web_limit: int = 15,
        auto_save: bool = True,
    ) -> dict[str, Any]:
        """
        Perform hybrid supplier discovery combining internal DB and web search.

        Args:
            project_id: Project to associate suppliers with
            query: Search query (will use intake context if not provided)
            l1_category: L1 category filter (will use intake context if not provided)
            l2_category: L2 category filter (will use intake context if not provided)
            region: Region filter (will use intake context if not provided)
            internal_limit: Max internal suppliers to find
            web_limit: Max web suppliers to find
            auto_save: Whether to automatically create ProjectSupplier records

        Returns:
            Combined results with deduplication
        """
        try:
            # Load project and intake context if query/category/region not provided
            async with get_db_context() as session:
                project_result = await session.execute(
                    select(Project).where(Project.id == project_id)
                )
                project = project_result.scalar_one_or_none()

                if not project:
                    return {
                        "error": "Project not found",
                        "ranked_suppliers": [],
                        "internal_suppliers": [],
                        "web_suppliers": [],
                        "total_count": 0,
                    }

                # Build search query from intake context if not provided
                if not query:
                    query_parts = []
                    if project.general_information:
                        query_parts.append(project.general_information[:200])
                    if project.metadata_:
                        if project.metadata_.get("category"):
                            query_parts.append(project.metadata_["category"])
                        if project.metadata_.get("subcategory"):
                            query_parts.append(project.metadata_["subcategory"])
                    query = " ".join(query_parts) if query_parts else "general business services"

                # Get L1/L2 category and region from project metadata if not provided
                if not l1_category and project.metadata_:
                    l1_category = project.metadata_.get("category")
                if not l2_category and project.metadata_:
                    l2_category = project.metadata_.get("subcategory")
                if not region and project.metadata_:
                    region = project.metadata_.get("region")

            # Run internal and web searches in parallel (conceptually - Python async)
            internal_result = await self.search_internal(
                query=query,
                l1_category=l1_category,
                l2_category=l2_category,
                region=region,
                limit=internal_limit,
            )

            web_result = await self.search_web(
                query=query,
                category=l1_category,  # Use L1 category for web search context
                region=region,
                limit=web_limit,
            )

            # Extract suppliers from structured results (handles both old list format and new dict format)
            internal_results = internal_result.get("suppliers", []) if isinstance(internal_result, dict) else internal_result
            web_results = web_result.get("suppliers", []) if isinstance(web_result, dict) else web_result

            # Collect search errors for error context
            search_errors = []
            if isinstance(internal_result, dict) and internal_result.get("error"):
                search_errors.append(f"Internal search: {internal_result['error']}")
            if isinstance(web_result, dict) and web_result.get("error"):
                search_errors.append(f"Web search: {web_result['error']}")

            # Deduplicate by supplier name (case-insensitive)
            seen_names: set[str] = set()
            deduplicated_internal = []
            deduplicated_web = []

            # Internal suppliers already have verified=True and boosted_score from search_internal
            for supplier in internal_results:
                name_key = supplier["name"].lower().strip()
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    deduplicated_internal.append(supplier)

            # Add scoring and verification status to web suppliers
            for idx, supplier in enumerate(web_results):
                name_key = supplier["name"].lower().strip()
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    # Calculate score for web suppliers based on position
                    # First result gets 0.85, decreasing by 0.05 for each subsequent result
                    web_score = max(0.30, 0.85 - (idx * 0.05))
                    if web_score < 0.45:
                        continue  # Skip weak web matches
                    supplier["similarity"] = web_score
                    supplier["boosted_score"] = web_score  # No boost for web suppliers
                    supplier["verified"] = False  # Web suppliers are not pre-verified

                    # Calculate tier based on score
                    if web_score >= 0.80:
                        supplier["tier"] = "A"
                    elif web_score >= 0.70:
                        supplier["tier"] = "B"
                    else:
                        supplier["tier"] = "C"

                    deduplicated_web.append(supplier)

            # Create merged ranked list - sorted by boosted_score (internal suppliers rank higher)
            ranked_suppliers = sorted(
                deduplicated_internal + deduplicated_web,
                key=lambda s: s.get("boosted_score", 0),
                reverse=True,
            )

            # Create ProjectSupplier records if auto_save is enabled
            created_records = []
            if auto_save:
                created_records = await self._save_project_suppliers(
                    project_id=project_id,
                    internal_suppliers=deduplicated_internal,
                    web_suppliers=deduplicated_web,
                    discovery_context={
                        "search_query": query,
                        "l1_category": l1_category,
                        "l2_category": l2_category,
                        "intake_region": region,
                    },
                )

            logger.info(
                "Hybrid search completed",
                extra={
                    "project_id": str(project_id),
                    "internal_count": len(deduplicated_internal),
                    "web_count": len(deduplicated_web),
                    "saved_count": len(created_records),
                }
            )

            return {
                "project_id": str(project_id),
                "ranked_suppliers": ranked_suppliers,  # Combined list sorted by score (internal prioritized)
                "internal_suppliers": deduplicated_internal,  # Kept for backward compatibility
                "web_suppliers": deduplicated_web,  # Kept for backward compatibility
                "total_count": len(deduplicated_internal) + len(deduplicated_web),
                "search_context": {
                    "query": query,
                    "l1_category": l1_category,
                    "l2_category": l2_category,
                    "region": region,
                },
                "created_records": created_records,
                "search_errors": search_errors if search_errors else None,  # Error info for caller
            }

        except Exception as e:
            logger.error(f"Hybrid search error: {e}", exc_info=True)
            return {
                "error": str(e),
                "ranked_suppliers": [],
                "internal_suppliers": [],
                "web_suppliers": [],
                "total_count": 0,
            }

    async def _save_project_suppliers(
        self,
        project_id: uuid.UUID,
        internal_suppliers: list[dict[str, Any]],
        web_suppliers: list[dict[str, Any]],
        discovery_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Create ProjectSupplier records for discovered suppliers.

        Uses centralized deduplication to prevent duplicate entries.

        Args:
            project_id: Project ID
            internal_suppliers: Suppliers from internal DB
            web_suppliers: Suppliers from web discovery
            discovery_context: Context about the discovery

        Returns:
            List of created record summaries
        """
        created = []

        async with get_db_context() as session:
            # Load existing suppliers ONCE at start for efficient deduplication
            existing_supplier_ids, existing_web_names = await get_existing_suppliers_for_project(
                session, project_id
            )

            # Track names added within this batch to prevent intra-batch duplicates
            batch_added_web_names: set[str] = set()
            skipped_internal = 0
            skipped_web = 0

            # Process internal suppliers
            for supplier in internal_suppliers:
                try:
                    supplier_uuid = uuid.UUID(supplier["id"])

                    # Check if already exists using the preloaded set
                    if supplier_uuid in existing_supplier_ids:
                        skipped_internal += 1
                        continue

                    record = ProjectSupplier(
                        project_id=project_id,
                        supplier_id=supplier_uuid,
                        status="suggested",
                        match_score=supplier.get("similarity"),
                        source="internal",
                        discovery_context={
                            **discovery_context,
                            "match_reason": f"Vector similarity: {supplier.get('similarity', 0):.2%}",
                        },
                        metadata_={
                            "tier": supplier.get("tier"),
                            "discovered_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    session.add(record)

                    # Add to existing set to prevent duplicates within this batch
                    existing_supplier_ids.add(supplier_uuid)

                    created.append({
                        "id": str(record.id),
                        "supplier_id": str(supplier_uuid),
                        "name": supplier["name"],
                        "source": "internal",
                        "match_score": supplier.get("similarity"),
                        "tier": supplier.get("tier"),
                    })
                except Exception as e:
                    logger.warning(f"Failed to save internal supplier: {e}")

            # Process web suppliers
            # Web suppliers get a default match score based on position (first = more relevant)
            for idx, supplier in enumerate(web_suppliers):
                try:
                    supplier_name = supplier.get("name", "")
                    normalized_name = normalize_supplier_name(supplier_name)

                    # Check if supplier already exists (by normalized name)
                    if normalized_name in existing_web_names:
                        skipped_web += 1
                        continue

                    # Check if already added in this batch
                    if normalized_name in batch_added_web_names:
                        skipped_web += 1
                        continue

                    # Calculate match score for web suppliers based on position
                    # First result gets 0.85, decreasing by 0.05 for each subsequent result
                    web_match_score = max(0.60, 0.85 - (idx * 0.05))

                    # Determine tier based on score
                    if web_match_score >= 0.80:
                        web_tier = "A"
                    elif web_match_score >= 0.70:
                        web_tier = "B"
                    else:
                        web_tier = "C"

                    record = ProjectSupplier(
                        project_id=project_id,
                        supplier_id=None,  # Web-discovered supplier
                        status="suggested",
                        match_score=web_match_score,  # Assign estimated score
                        source="web_discovery",
                        external_supplier_data={
                            "name": supplier_name,
                            "description": supplier.get("description", ""),
                            "website": supplier.get("website", ""),
                            "location": supplier.get("location", ""),
                            "capabilities": supplier.get("capabilities", []),
                            "certifications": supplier.get("certifications", []),
                            "source_urls": supplier.get("source_urls", []),
                            "employee_count": supplier.get("employee_count"),
                            "revenue": supplier.get("revenue"),
                            "coverage": supplier.get("coverage", []),
                            "products_and_services": supplier.get("products_and_services") or supplier.get("capabilities", []),
                        },
                        discovery_context={
                            **discovery_context,
                            "match_reason": "Web discovery via Amazon Nova",
                        },
                        metadata_={
                            "tier": web_tier,
                            "discovered_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    session.add(record)

                    # Track for intra-batch deduplication
                    batch_added_web_names.add(normalized_name)
                    existing_web_names.add(normalized_name)

                    created.append({
                        "id": str(record.id),
                        "supplier_id": None,
                        "name": supplier_name,
                        "source": "web_discovery",
                        "match_score": web_match_score,
                        "tier": web_tier,
                    })
                except Exception as e:
                    logger.warning(f"Failed to save web supplier: {e}")

            await session.commit()

            if skipped_internal > 0 or skipped_web > 0:
                logger.info(
                    "Skipped duplicate suppliers during save",
                    extra={
                        "project_id": str(project_id),
                        "skipped_internal": skipped_internal,
                        "skipped_web": skipped_web,
                        "created_count": len(created),
                    }
                )

        return created

    async def add_supplier_to_project(
        self,
        project_id: uuid.UUID,
        supplier_data: dict[str, Any],
        source: str = "manual",
    ) -> dict[str, Any]:
        """
        Add a supplier to a project (for chat-based additions).

        Includes duplicate checking - returns error if supplier already exists.

        Args:
            project_id: Project ID
            supplier_data: Supplier data (either with supplier_id or external data)
            source: Source of addition ("manual", "chat", "web_discovery")

        Returns:
            Created ProjectSupplier record info, or error dict if duplicate
        """
        async with get_db_context() as session:
            supplier_id = supplier_data.get("supplier_id")

            # Load existing suppliers for duplicate checking
            existing_supplier_ids, existing_web_names = await get_existing_suppliers_for_project(
                session, project_id
            )

            if supplier_id:
                # Internal supplier - check for duplicate by UUID
                supplier_uuid = uuid.UUID(supplier_id)

                if supplier_uuid in existing_supplier_ids:
                    logger.info(
                        "Duplicate internal supplier rejected",
                        extra={
                            "project_id": str(project_id),
                            "supplier_id": supplier_id,
                            "source": source,
                        }
                    )
                    return {
                        "duplicate": True,
                        "error": "This supplier has already been added to the project",
                        "supplier_id": supplier_id,
                    }

                record = ProjectSupplier(
                    project_id=project_id,
                    supplier_id=supplier_uuid,
                    status="suggested",
                    source=source,
                    discovery_context={"match_reason": f"Added via {source}"},
                    metadata_={"added_at": datetime.now(timezone.utc).isoformat()},
                )
            else:
                # Web/external supplier - check for duplicate by normalized name
                supplier_name = supplier_data.get("name", "Unknown")
                normalized_name = normalize_supplier_name(supplier_name)

                if normalized_name and normalized_name in existing_web_names:
                    logger.info(
                        "Duplicate web supplier rejected",
                        extra={
                            "project_id": str(project_id),
                            "supplier_name": supplier_name,
                            "normalized_name": normalized_name,
                            "source": source,
                        }
                    )
                    return {
                        "duplicate": True,
                        "error": f"A supplier with a similar name ('{supplier_name}') has already been added to the project",
                        "name": supplier_name,
                    }

                record = ProjectSupplier(
                    project_id=project_id,
                    supplier_id=None,
                    status="suggested",
                    source=source,
                    external_supplier_data={
                        "name": supplier_name,
                        "description": supplier_data.get("description", ""),
                        "website": supplier_data.get("website", ""),
                        "location": supplier_data.get("location", ""),
                        "capabilities": supplier_data.get("capabilities", []),
                    },
                    discovery_context={"match_reason": f"Added via {source}"},
                    metadata_={"added_at": datetime.now(timezone.utc).isoformat()},
                )

            session.add(record)
            await session.commit()
            await session.refresh(record)

            return {
                "id": str(record.id),
                "project_id": str(project_id),
                "supplier_id": str(record.supplier_id) if record.supplier_id else None,
                "name": record.supplier_name,
                "source": record.source,
                "status": record.status,
            }

    async def remove_supplier_from_project(
        self,
        project_supplier_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Remove a supplier from a project.

        Args:
            project_supplier_id: ProjectSupplier record ID

        Returns:
            Result of deletion
        """
        async with get_db_context() as session:
            result = await session.execute(
                select(ProjectSupplier).where(ProjectSupplier.id == project_supplier_id)
            )
            record = result.scalar_one_or_none()

            if not record:
                return {"error": "ProjectSupplier not found", "deleted": False}

            supplier_name = record.supplier_name
            await session.delete(record)
            await session.commit()

            return {
                "id": str(project_supplier_id),
                "name": supplier_name,
                "deleted": True,
            }

    async def update_supplier_status(
        self,
        project_supplier_id: uuid.UUID,
        status: str,
    ) -> dict[str, Any]:
        """
        Update the status of a ProjectSupplier.

        Args:
            project_supplier_id: ProjectSupplier record ID
            status: New status ("suggested", "shortlisted", "contacted", "rejected", "selected")

        Returns:
            Updated record info
        """
        valid_statuses = {"suggested", "shortlisted", "contacted", "rejected", "selected"}
        if status not in valid_statuses:
            return {"error": f"Invalid status. Must be one of: {valid_statuses}"}

        async with get_db_context() as session:
            result = await session.execute(
                select(ProjectSupplier).where(ProjectSupplier.id == project_supplier_id)
            )
            record = result.scalar_one_or_none()

            if not record:
                return {"error": "ProjectSupplier not found"}

            record.status = status
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()

            return {
                "id": str(project_supplier_id),
                "name": record.supplier_name,
                "status": status,
                "updated": True,
            }

    async def promote_to_internal(
        self,
        project_supplier_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Promote a web-discovered supplier to the internal Supplier database.

        Args:
            project_supplier_id: ProjectSupplier record ID

        Returns:
            Created internal Supplier info
        """
        async with get_db_context() as session:
            result = await session.execute(
                select(ProjectSupplier).where(ProjectSupplier.id == project_supplier_id)
            )
            record = result.scalar_one_or_none()

            if not record:
                return {"error": "ProjectSupplier not found"}

            if not record.is_web_discovered:
                return {"error": "Supplier is already in internal database"}

            external_data = record.external_supplier_data or {}

            # Create new internal Supplier
            description = external_data.get("description", "")
            name = external_data.get("name", "Unknown")

            # Generate embedding for the new supplier
            embedding = await self.embedding_service.embed(f"{name} {description}")

            new_supplier = Supplier(
                name=name,
                description=description,
                categories=external_data.get("capabilities", []),
                capabilities={
                    "certifications": external_data.get("certifications", []),
                    "services": external_data.get("capabilities", []),
                },
                contact_info={
                    "website": external_data.get("website", ""),
                    "location": external_data.get("location", ""),
                },
                metadata_={
                    "promoted_from_web": True,
                    "original_source_urls": external_data.get("source_urls", []),
                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                },
                embedding=embedding,
                is_active=True,
            )

            session.add(new_supplier)
            await session.flush()

            # Update ProjectSupplier to point to new internal supplier
            record.supplier_id = new_supplier.id
            record.external_supplier_data = None
            record.source = "internal"  # Update source since it's now internal
            record.metadata_ = {
                **(record.metadata_ or {}),
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            }

            await session.commit()

            return {
                "supplier_id": str(new_supplier.id),
                "project_supplier_id": str(project_supplier_id),
                "name": new_supplier.name,
                "promoted": True,
            }


# Singleton instance
_hybrid_search_service: HybridSupplierSearchService | None = None


def get_hybrid_search_service() -> HybridSupplierSearchService:
    """Get the singleton HybridSupplierSearchService instance."""
    global _hybrid_search_service
    if _hybrid_search_service is None:
        _hybrid_search_service = HybridSupplierSearchService()
    return _hybrid_search_service

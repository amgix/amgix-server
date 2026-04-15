from typing import Dict, List, Any, Optional, Tuple, Union
import re
import asyncio
import asyncpg
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from .sql_base import SQLBase, TransactionRollback
from ..models.document import Document, DocumentWithVectors, SearchResult
from ..models.vector import CollectionConfigInternal, SearchQueryWithVectors, VectorData
from ..common import (
    SEARCH_PREFETCH_MULTIPLIER, DatabaseFeatures, DatabaseInfo, APP_PREFIX,
    DEFAULT_DB_POOL_SIZE, MIN_DB_POOL_SIZE
)


class PostgreSQLDatabase(SQLBase):
    """
    PostgreSQL implementation of the SQLBase interface.
    
    This class overrides SQL templates to provide PostgreSQL-specific syntax.
    All business logic is inherited from the base class.
    """
    
    def __init__(self, connection_string: str, logger, **kwargs):
        """Initialize PostgreSQL connection parameters."""
        super().__init__(connection_string, logger=logger, **kwargs)
        
        # Override SQL templates with PostgreSQL-specific syntax (merge with base templates)
        pgsql_templates = {
            # Table Options and Feature Flags
            "table_options": "",
            "if_not_exists": " IF NOT EXISTS ",
            
            # Identifier Quoting (PostgreSQL uses double quotes)
            "quote_char": '"',
            "quote_identifier": '"{identifier}"',
            
            # Generic Table Creation Template
            "create_table": """
                CREATE TABLE {if_not_exists}"{table_name}" (
                    {columns}{indexes}
                ){table_options}
            """,
            
            # Column Definition Templates (with double quotes for PostgreSQL)
            "column_pk": '"{name}" BIGSERIAL PRIMARY KEY',
            "column_varchar": '"{name}" VARCHAR({size}){null_constraint}',
            "column_text": '"{name}" TEXT{null_constraint}',
            "column_longtext": '"{name}" TEXT{null_constraint}',
            "column_int": '"{name}" INTEGER{null_constraint}',
            "column_smallint": '"{name}" SMALLINT{null_constraint}',
            "column_bigint": '"{name}" BIGINT{null_constraint}',
            "column_float": '"{name}" REAL{null_constraint}',
            "column_decimal": '"{name}" NUMERIC(38,10){null_constraint}',
            "column_timestamp": '"{name}" TIMESTAMP WITH TIME ZONE{default}',
            "current_timestamp": " DEFAULT NOW()",

            "column_vector": '"{name}" VECTOR({dimensions}) NOT NULL',
            
            # Index Definition Templates (with double quotes for PostgreSQL)
            "index_primary": 'CONSTRAINT "pk_{name}" PRIMARY KEY ("{column}")',
            "index_primary_multi": 'CONSTRAINT "pk_{name}" PRIMARY KEY ({columns})',
            "index_unique": 'CONSTRAINT "ix_{name}" UNIQUE ({columns})',
            "index_simple": 'INDEX "ix_{name}" ({columns})',
            "index_foreign_key": 'CONSTRAINT "fk_{name}" FOREIGN KEY ({columns}) REFERENCES "{ref_table}"({ref_columns}){on_delete}',
            "create_index_vector_cosine": 'CREATE INDEX "ix_{name}" ON "{table}" USING hnsw ("{column}" vector_cosine_ops) WITH (m=16,ef_construction=64)',
            "create_index_vector_dot": 'CREATE INDEX "ix_{name}" ON "{table}" USING hnsw ("{column}" vector_ip_ops) WITH (m=16,ef_construction=64)',
            "create_index_vector_euclid": 'CREATE INDEX "ix_{name}" ON "{table}" USING hnsw ("{column}" vector_l2_ops) WITH (m=16,ef_construction=64)',
            # Separate index creation for simple indexes
            "create_index_simple": 'CREATE INDEX "ix_{name}" ON "{table}" ({columns})',
            "create_unique_index": 'CREATE UNIQUE INDEX "ix_{name}" ON "{table}" ({columns})',
            # Vector search session tuning
            "ef_search": 'SET LOCAL hnsw.ef_search = {ef_search}',
            
            # Query Templates (with double quotes for PostgreSQL)
            "insert": 'INSERT INTO "{table}" ({columns}) VALUES ({placeholders}){get_last_id}',
            "batch_insert": 'INSERT INTO "{table}" ({columns}) VALUES {placeholders}',
            "update": 'UPDATE "{table}" SET {set_clause} WHERE {where_clause}',
            "delete": 'DELETE FROM "{table}" WHERE {where_clause}',
            "select": 'SELECT {columns} FROM "{table}"{joins}{where}{group_by}{order_by}{limit}',
            "count": 'SELECT COUNT(*) as count FROM "{table}"{where}',
            "drop_table": "DROP TABLE IF EXISTS {tables}",
            "alter_table_add_index": 'ALTER TABLE "{table}" ADD {index_definition}',
            
            # Upsert/Join/Delete templates with double-quoted table identifiers
            "upsert": 'INSERT INTO "{table}" ({columns}) VALUES {values} ON CONFLICT ({conflict_columns}) DO UPDATE SET {update_clause}',
            # Update clause for IDF upsert (PostgreSQL)
            "upsert_update_doc_count": '"doc_count" = "{table}"."doc_count" + EXCLUDED."doc_count"',
            "upsert_update_metric_bucket": '"value" = EXCLUDED."value", "n" = EXCLUDED."n"',
            "update_join": 'UPDATE "{table}" {alias} SET {set_clause} FROM ({subquery}) {subquery_alias} WHERE {join_conditions}',
            "delete_join": 'DELETE FROM "{table}" {alias} USING ({subquery}) {subquery_alias} WHERE {join_conditions} AND {where_clause}',
            
            # Transaction Templates
            "last_insert_id": " RETURNING pk_id as pk_id",
            
            # Temp tables and maintenance with double-quoted table identifiers
            "create_temp_table": 'CREATE TEMPORARY TABLE IF NOT EXISTS "{table_name}" ({columns}){table_options}',
            "truncate": 'TRUNCATE "{table}"',
            
            # Vector Search Templates (with double quotes for PostgreSQL)
            "dense_distance_cosine": '("{field_name}" <=> %({param_name})s)',
            "dense_distance_dot": '("{field_name}" <#> %({param_name})s)',
            "dense_distance_euclid": '("{field_name}" <-> %({param_name})s)',
            "dense_vector_insert": '{vector_values}',
            
            # IDF Filter Template (PostgreSQL-specific with double quotes)
            "idf_filter": 'AND idf."doc_count" <= %(idf_threshold)s',
            "query_vector_batch_select_first": """
                SELECT
                    CAST(%s AS SMALLINT) AS field_vector_id,
                    CAST(%s AS BIGINT) AS token_id,
                    CAST(%s AS REAL) AS weight,
                    CAST(%s AS INTEGER) AS requires_idf
            """,
            "query_vector_batch_select_next": "SELECT %s, %s, %s, %s",
            "query_vector_weighted_insert": """
                INSERT INTO {table} ({field_vector_col}, {token_col}, {weight_col})
                SELECT src.field_vector_id, src.token_id,
                    CASE
                        WHEN src.requires_idf = 1 THEN src.weight * {idf_expr}
                        ELSE src.weight
                    END AS weight
                FROM ({source_rows}) src
                LEFT JOIN {idf_table} idf
                    ON idf.{idf_field_vector_col} = src.field_vector_id
                    AND idf.{idf_token_col} = src.token_id
                CROSS JOIN (SELECT COUNT(*) AS total_docs FROM {docs_table}) doc_stats
                WHERE (src.requires_idf = 0 OR idf.{idf_token_col} IS NOT NULL)
            """,
            
            "hybrid_search": """
                SELECT vds.fv_id, d."id", d."name", d."description", d."timestamp", d."metadata", 
                    (SELECT string_agg(t."tag", '|') FROM "{tags_table}" t WHERE t."doc_pk_id" = d."pk_id") tags,
                    vds.vdscore as score
                FROM "{documents_table}" d
                INNER JOIN (
                    SELECT fv_id, "doc_pk_id", vdscore 
                    FROM (
                        {all_unions}
                    ) combined_vectors
                ) vds ON vds."doc_pk_id" = d."pk_id"
                ORDER BY vds.fv_id, score DESC
            """,
            
            "hybrid_scores_only": """
                SELECT fv_id, "doc_pk_id", vdscore 
                FROM (
                    {all_unions}
                ) combined_vectors
            """,
            
            # Select documents by pk_id list with aggregated tags (PostgreSQL uses string_agg)
            "select_docs_in_with_tags": """
                SELECT 
                    d.{pk_col} AS pk_id,
                    d.{id_col} AS id,
                    d.{name_col} AS name,
                    d.{description_col} AS description,
                    d.{timestamp_col} AS timestamp,
                    d.{metadata_col} AS metadata,
                    (SELECT string_agg(t.{tag_col}, '|')
                       FROM {tags_table} t
                      WHERE t.{tag_doc_pk_col} = d.{pk_col}) AS tags
                FROM {table} d
                WHERE d.{pk_col} IN ({placeholders})
            """,
            
            # Database Probing Templates (with double quotes for PostgreSQL)
            "version_query": "SELECT current_setting('server_version') AS version",
            "vector_test_create": f'CREATE TEMPORARY TABLE "{APP_PREFIX}_test_vector_idx" (v VECTOR(1) NOT NULL)',
            "vector_test_drop": f'DROP TABLE "{APP_PREFIX}_test_vector_idx"',
            
            # Document Type Filtering Templates (with double quotes for PostgreSQL)
            "tags_filter": 'WHERE d.pk_id IN (SELECT doc_pk_id FROM "{tags_table}" t WHERE t.doc_pk_id = d.pk_id AND t.tag = ANY(%(document_tags)s))',
            "tags_filter_and": 'WHERE d.pk_id IN (SELECT doc_pk_id FROM "{tags_table}" t WHERE t.doc_pk_id = d.pk_id AND t.tag = ANY(%(document_tags)s) GROUP BY doc_pk_id HAVING COUNT(DISTINCT t.tag) = %(document_tags_count)s)',
            "sparse_documents_join": 'INNER JOIN "{documents_table}" d ON d."pk_id" = vd."doc_pk_id"',
            "sparse_idf": "ln((%(total_docs)s + 1) / (idf.doc_count + 0.5))",
            
            # Sparse Vector Search Templates (with double quotes for PostgreSQL)
            "sparse_union_single": """
                SELECT {field_vector_id} fv_id, "doc_pk_id", vdscore FROM (
                    SELECT {field_vector_id} fv_id, vd."doc_pk_id", sum(vd."weight" * qv."weight" * {sparse_idf}) vdscore
                      FROM "{vector_data_table}" vd
                      {sparse_documents_join}
                      INNER JOIN "{query_vectors_table}" qv ON qv."field_vector_id" = vd."field_vector_id" 
                        AND qv."token_id" = vd."token_id" 
                        AND qv."field_vector_id" = {field_vector_id}
                      {idf_join}
                      {tags_filter}
                      {idf_filter}
                      GROUP BY vd."doc_pk_id"
                      ORDER BY vdscore DESC 
                      LIMIT %(prefetch_limit)s
                ) sparse_subquery""",
            
            # Dense Vector Search Templates (with double quotes for PostgreSQL)
            "dense_union_single": """
                SELECT {field_vector_id} fv_id, "pk_id" "doc_pk_id", vdscore FROM (
                    SELECT {field_vector_id} fv_id, d."pk_id", (1 - {distance_expr}) vdscore
                      FROM "{documents_table}" d 
                      {tags_filter}
                      ORDER BY {distance_expr} 
                      LIMIT %(prefetch_limit)s
                ) dense_subquery""",
        }
        
        # Merge PostgreSQL-specific templates with base templates
        self.SQL_TEMPLATES.update(pgsql_templates)
        
        # Parse connection string using urllib.parse
        parsed = urlparse(connection_string)
        
        # Extract components
        self.host = parsed.hostname
        self.port = parsed.port or 5432
        self.user = parsed.username or ''
        self.password = parsed.password or ''
        self.database = parsed.path.lstrip('/') if parsed.path else ''
        
        # Validate required components
        if not self.host:
            raise ValueError("Host is required in connection string")
        if not self.database:
            raise ValueError("Database name is required in connection string")
        
        # Connection pool
        self.pool = None

    async def _get_connection(self):
        """Get a connection from the pool."""
        if self.pool is None:
            # Lazy-connect to avoid requiring explicit probe/connect in callers
            await self.connect()
        return await self.pool.acquire()

    async def connect(self):
        """Establish connection pool to the database."""
        async def init_connection(conn):
            # Set transaction isolation level to READ COMMITTED to reduce deadlocks
            await conn.execute('SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED')
        
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                min_size=MIN_DB_POOL_SIZE,
                max_size=DEFAULT_DB_POOL_SIZE,
                init=init_connection
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    async def create_collection(self, collection_name: str, config: CollectionConfigInternal) -> bool:
        """Create collection via base implementation; cleanup if DDL partially applied.

        PostgreSQL transactions can rollback DDL, but for consistency with MariaDB
        and to ensure a clean state, we will still attempt a cleanup.
        """
        try:
            return await super().create_collection(collection_name, config)
        except Exception:
            # Cleanup best-effort: drop any partially created tables, then re-raise
            try:
                await self.delete_collection(collection_name)
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to cleanup collection {collection_name} after creation failure: {cleanup_error}")
            raise

    async def execute_sql(self, sql: str, params: Optional[Union[tuple, list, dict]] = None, skip_first: bool = False, conn=None) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dictionaries."""
        # Use provided connection or get one from the pool
        connection_provided = conn is not None
        if not connection_provided:
            conn = await self._get_connection()
        
        try:
            # asyncpg doesn't have cursor concept, directly execute
            # skip_first is used by MariaDB for multi-statement results (LAST_INSERT_ID)
            # For PostgreSQL we use RETURNING instead, so skip_first is not needed
            # Convert %s/%(name)s placeholders to $1.. and map params to positional list
            pg_sql, pg_params = self._convert_placeholders(sql, params)
            records = await conn.fetch(pg_sql, *pg_params)
            return [dict(r) for r in records]
        finally:
            # Only release if we got the connection ourselves
            if not connection_provided:
                await self.pool.release(conn)

    async def execute_sql_no_result(self, sql: str, params: Optional[Union[tuple, list, dict]] = None, conn=None) -> None:
        """Execute SQL that doesn't return results (INSERT, UPDATE, DELETE)."""
        # Use provided connection or get one from the pool
        connection_provided = conn is not None
        if not connection_provided:
            conn = await self._get_connection()
        
        try:
            # Convert %s/%(name)s placeholders to $1.. and map params to positional list
            pg_sql, pg_params = self._convert_placeholders(sql, params)
            await conn.execute(pg_sql, *pg_params)
        finally:
            # Only release if we got the connection ourselves
            if not connection_provided:
                await self.pool.release(conn)

    # ==========================================
    # Transaction Management (PostgreSQL)
    # ==========================================
    @asynccontextmanager
    async def transaction(self):
        """Context manager for transactions that ensures proper commit/rollback."""
        conn = await self._get_connection()
        try:
            # Start transaction (asyncpg equivalent of disabling autocommit)
            tr = conn.transaction()
            await tr.start()
            
            # Yield the connection to be used in the context block
            yield conn
            
            # If we get here without exceptions, commit the transaction
            await tr.commit()
        except Exception as e:
            # On any error, rollback the transaction
            await tr.rollback()
            if isinstance(e, TransactionRollback):
                # Do not re-raise for controlled rollback-on-success
                return
            raise
        finally:
            # Always return the connection to the pool
            await self.pool.release(conn)

    def _convert_placeholders(self, sql: str, params: Optional[Union[tuple, list, dict]]) -> Tuple[str, List[Any]]:
        """Convert %s and %(name)s placeholders to $1.. and return positional params.

        - If params is a dict, replace named placeholders in order of appearance and build
          a positional list using those names. Repeated names duplicate the value.
        - Then replace any remaining bare %s using remaining positional params (if provided),
          or leave none if there are no more.
        - If params is a sequence, only process bare %s in order.
        """
        if params is None:
            return sql, []

        positional: List[Any] = []
        index = 0
        out_sql = sql

        # Handle named placeholders first if params is a dict
        if isinstance(params, dict):
            def repl_named(m: re.Match) -> str:
                nonlocal index, positional
                name = m.group(1)
                index += 1
                positional.append(params[name])
                return f"${index}"

            out_sql = re.sub(r"%\(([^)]+)\)s", repl_named, out_sql)

            # No remaining positional params expected for dicts; but if the SQL also contains %s
            # we cannot fulfill them from a dict, so leave as-is (will error) unless caller passed both.
            # To support mixed usage with an additional sequence, allow a tuple/list in a tuple form (dict, seq).
            remaining_seq: List[Any] = []
        else:
            remaining_seq = list(params)

        # Replace bare %s using remaining_seq
        def repl_positional(_: re.Match) -> str:
            nonlocal index, positional, remaining_seq
            index += 1
            if not remaining_seq:
                # No param available - still replace to keep SQL valid; asyncpg will error on arg count
                positional.append(None)
            else:
                positional.append(remaining_seq.pop(0))
            return f"${index}"

        out_sql = re.sub(r"%s", repl_positional, out_sql)

        return out_sql, positional

    async def configure(self) -> None:
        await self.execute_sql_no_result("CREATE EXTENSION IF NOT EXISTS vector")
        await super().configure()

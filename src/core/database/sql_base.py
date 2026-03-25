import copy
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from enum import Enum
from collections import defaultdict
import uuid
import logging
import asyncio
import json
from datetime import datetime, timezone
from abc import abstractmethod
from contextlib import asynccontextmanager
from .base import DatabaseBase, AmgixNotFound
from .common import AmgixValidationError
from ..models.document import Document, DocumentWithVectors, QueueDocument, QueueInfo, SearchResult, DocumentStatus, DocumentStatusResponse, VectorScore
from ..models.vector import CollectionConfigInternal, SearchQueryWithVectors, CollectionConfig, VectorConfig, MetadataFilter, internal_to_user_config
from ..common import (
    VectorType, DenseDistance, APP_PREFIX, DatabaseInfo, DatabaseFeatures, 
    SEARCH_PREFETCH_MULTIPLIER, IDF_THRESHOLD_MULTIPLIER, DEFAULT_SQL_BATCH_SIZE,
    MAX_METADATA_KEY_LENGTH, MAX_METADATA_VALUE_LENGTH, UUID_LENGTH, MAX_DOCUMENT_TAG_LENGTH,
    MAX_FIELD_VECTOR_NAME_LENGTH, MAX_INTERNAL_COLLECTION_NAME_LENGTH,
    MAX_DOCUMENT_ID_LENGTH, QueuedDocumentStatus, MAX_STATUS_LENGTH, MAX_SEARCH_LIMIT,
    get_user_collection_name, MetadataValueType, COLLECTION_INGEST_LOCK_TIMEOUT
)
from ..common.lock_manager import LockClient


class TransactionRollback(Exception):
    pass


class SQLBase(DatabaseBase):
    """
    Abstract base class for SQL database implementations.
    
    Provides a template-based approach to SQL generation with a dictionary
    of SQL snippets that can be overridden by derived classes.
    """
    
    def __init__(self, connection_string: str, logger, **kwargs):
        super().__init__(connection_string, logger=logger, **kwargs)
        # Cache quote_char for performance
        self._quote_char = self.SQL_TEMPLATES.get("quote_char", '"')
    
    class TableType(Enum):
        """Enumeration of table types for collections."""
        DOCUMENTS = "docs"
        VECTOR_DATA = "vectors"
        QUERY_VECTORS = "query"
        TAGS = "tags"
        IDF = "idf"
    
    def get_table_name(self, collection_name: str, table_type: 'SQLBase.TableType') -> str:
        """
        Generate a table name for a collection and table type.
        
        Args:
            collection_name: Name of the collection (ignored for system tables)
            table_type: Type of table (meta, documents, vector_data, query_vectors)
            
        Returns:
            Full table name in the format: {collection_name}_{table_type} for collection tables,
            or {APP_PREFIX}sys_{table_type} for system tables
        """
        if table_type == self.TableType.QUERY_VECTORS:
            return f"{APP_PREFIX}_sys_{table_type.value}"
        return f"{APP_PREFIX}_{self._string_to_uuid(collection_name)}_{table_type.value}"
    
    # Configuration
    DEFAULT_BATCH_SIZE = DEFAULT_SQL_BATCH_SIZE  # Number of rows to insert in one batch
    
    # Connection parameters (to be set by derived classes)
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = ""
    
    # SQL Templates Dictionary
    # This dictionary contains all SQL templates used by the database implementation
    # Derived classes should override these templates as needed
    SQL_TEMPLATES = {
        # Identifier Quoting
        "quote_char": '"',
        "quote_identifier": '"{identifier}"',
        
        # Generic Table Creation Template
        "create_table": """
            CREATE TABLE {if_not_exists}{table_name} (
                {columns}{indexes}
            ){table_options}
        """,
        
        # Column Definition Templates
        "column_pk": '"{name}" BIGINT AUTO_INCREMENT PRIMARY KEY',
        "column_varchar": '"{name}" VARCHAR({size}){null_constraint}',
        "column_text": '"{name}" TEXT{null_constraint}',
        "column_longtext": '"{name}" LONGTEXT{null_constraint}',
        "column_int": '"{name}" INT{null_constraint}',
        "column_smallint": '"{name}" SMALLINT{null_constraint}',
        "column_bigint": '"{name}" BIGINT{null_constraint}',
        "column_float": '"{name}" FLOAT{null_constraint}',
        "column_decimal": '"{name}" DECIMAL(38,10){null_constraint}',
        "column_timestamp": '"{name}" TIMESTAMP{default}',
        "current_timestamp": " DEFAULT CURRENT_TIMESTAMP",

        "column_vector": '"{name}" VECTOR({dimensions}){null_constraint}',
        
        # Index Definition Templates
        "index_primary": 'CONSTRAINT "pk_{name}" PRIMARY KEY ("{column}")',
        "index_primary_multi": 'CONSTRAINT "pk_{name}" PRIMARY KEY ({columns})',
        "index_unique": 'UNIQUE KEY "ix_{name}" ({columns})',
        "index_simple": 'INDEX "ix_{name}" ({columns})',
        "index_foreign_key": 'CONSTRAINT "fk_{name}" FOREIGN KEY ({columns}) REFERENCES {ref_table}({ref_columns}){on_delete}',
        "index_vector": 'INDEX "ix_{name}" ("{column}")',
        
        # Query Templates
        "insert": "INSERT INTO {table} ({columns}) VALUES ({placeholders});{get_last_id}",
        "batch_insert": "INSERT INTO {table} ({columns}) VALUES {placeholders}",
        "upsert": "INSERT INTO {table} ({columns}) VALUES {values} ON DUPLICATE KEY UPDATE {update_clause}",
        # Backend-specific update clause for IDF upsert; default works for MariaDB/MySQL
        "upsert_update_doc_count": "doc_count = doc_count + 1",
        "query_vector_batch_select_first": """
            SELECT
                %s AS field_vector_id,
                %s AS token_id,
                %s AS weight,
                %s AS requires_idf
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
        "update": "UPDATE {table} SET {set_clause} WHERE {where_clause}",
        "update_join": "UPDATE {table} {alias} JOIN ({subquery}) {subquery_alias} ON {join_conditions} SET {set_clause}",
        "delete_join": "DELETE {alias} FROM {table} {alias} JOIN ({subquery}) {subquery_alias} ON {join_conditions} WHERE {where_clause}",
        "delete": "DELETE FROM {table} WHERE {where_clause}",
        "select": "SELECT {columns} FROM {table}{joins}{where}{group_by}{order_by}{limit}",
        "count": "SELECT COUNT(*) as count FROM {table}{where}",
        "drop_table": "DROP TABLE IF EXISTS {tables}",
        "alter_table_add_index": "ALTER TABLE {table} ADD {index_definition}",
        # Simple and unique index creation (default; backends can override)
        "create_index_simple": 'CREATE INDEX "ix_{name}" ON {table} ({columns})',
        "create_unique_index": 'CREATE UNIQUE INDEX "ix_{name}" ON {table} ({columns})',
        # Vector index creation (default; backends can override)
        "create_index_vector_cosine": 'CREATE INDEX "ix_{name}" ON {table} ("{column}")',
        "create_index_vector_dot": 'CREATE INDEX "ix_{name}" ON {table} ("{column}")',
        "create_index_vector_euclid": 'CREATE INDEX "ix_{name}" ON {table} ("{column}")',
        # Vector search session tuning (default noop; backends override)
        "ef_search": "",
        # Temp tables and maintenance
        "create_temp_table": "CREATE TEMPORARY TABLE IF NOT EXISTS {table_name} ({columns}){table_options}",
        "truncate": "TRUNCATE {table}",
        
        "last_insert_id": "SELECT LAST_INSERT_ID() as pk_id",
        
        # Vector Search Templates
        "dense_distance_cosine": 'd.{field_name} <-> %({param_name})s',
        "dense_distance_dot": 'd.{field_name} <-> %({param_name})s',  # Same as cosine for PostgreSQL
        "dense_distance_euclid": 'd.{field_name} <-> %({param_name})s',  # Same as cosine for PostgreSQL
        "dense_vector_insert": 'Vec_FromText({vector_values})',
        "hybrid_search": """
            SELECT vds.fv_id, d.`id`, d.`name`, d.`description`, d.`timestamp`, d.`metadata`, 
                (SELECT GROUP_CONCAT(t.`tag` SEPARATOR '|') FROM `{tags_table}` t WHERE t.`doc_pk_id` = d.`pk_id`) tags,
                vds.vdscore as score
            FROM `{documents_table}` d
            INNER JOIN (
                SELECT fv_id, `doc_pk_id`, vdscore 
                FROM (
                    {all_unions}
                ) combined_vectors
            ) vds ON vds.`doc_pk_id` = d.`pk_id`
            ORDER BY vds.fv_id, score DESC
        """,
        
        # Scores-only wrapper for union arms (no join to documents)
        "hybrid_scores_only": """
            SELECT fv_id, `doc_pk_id`, vdscore 
            FROM (
                {all_unions}
            ) combined_vectors
        """,
        
        # Collection Management Templates
        "list_collections_query": """
            SELECT DISTINCT 
                SUBSTRING(table_name, 1, LENGTH(table_name) - {docs_suffix_length}) as collection_name
            FROM information_schema.tables 
            WHERE table_schema = '{database}' 
            AND table_name LIKE '{prefix}%' 
            AND table_name NOT LIKE '{prefix}sys_%' 
            AND table_name LIKE '%_{docs_suffix}'
        """,
        
        # Database Probing Templates
        "version_query": "SELECT VERSION() AS version",

        "vector_test_create": f"CREATE TEMPORARY TABLE {APP_PREFIX}_test_vector_idx (v VECTOR(1), VECTOR INDEX(v) USING HNSW WITH (M=8))",
        "vector_test_drop": f"DROP TEMPORARY TABLE {APP_PREFIX}_test_vector_idx",
        
        # Sparse Vector Search Templates
        "sparse_union_single": """
            SELECT {field_vector_id} fv_id, `doc_pk_id`, vdscore FROM (
                SELECT {field_vector_id} fv_id, vd.`doc_pk_id`, SUM(vd.`weight` * qv.`weight` * {sparse_idf}) vdscore
                    FROM `{vector_data_table}` vd
                    {sparse_documents_join}
                    INNER JOIN `{query_vectors_table}` qv ON qv.`field_vector_id` = vd.`field_vector_id` 
                    AND qv.`token_id` = vd.`token_id` 
                    AND qv.`field_vector_id` = {field_vector_id}
                    {idf_join}
                    {tags_filter}
                    {idf_filter}
                    GROUP BY vd.`doc_pk_id`
                    ORDER BY vdscore DESC 
                    LIMIT %(prefetch_limit)s
            ) sparse_subquery""",
        
        # IDF Join Template (SQL-agnostic; identifiers should be pre-quoted)
        "idf_join": "INNER JOIN {idf_table} idf ON idf.{token_col} = vd.{token_col} AND idf.{fv_col} = vd.{fv_col}",
        
        # IDF Filter Template (filters out high-DF tokens)
        "idf_filter": "AND idf.`doc_count` <= %(idf_threshold)s",
        
        # Dense Vector Search Templates
        "dense_union_single": """
            SELECT {field_vector_id} fv_id, `pk_id` `doc_pk_id`, vdscore FROM (
                SELECT {field_vector_id} fv_id, d.`pk_id`, (1 - {distance_expr}) vdscore
                    FROM `{documents_table}` d 
                    {tags_filter}
                    ORDER BY {distance_expr} 
                    LIMIT %(prefetch_limit)s
            ) dense_subquery""",
        
        # Tags Filtering Templates
        "tags_filter": "WHERE d.pk_id IN (SELECT doc_pk_id FROM {tags_table} t WHERE t.doc_pk_id = d.pk_id AND t.tag IN %(document_tags)s)",
        "tags_filter_and": "WHERE d.pk_id IN (SELECT doc_pk_id FROM {tags_table} t WHERE t.doc_pk_id = d.pk_id AND t.tag IN %(document_tags)s GROUP BY doc_pk_id HAVING COUNT(DISTINCT t.tag) = %(document_tags_count)s)",
        "sparse_documents_join": "INNER JOIN {documents_table} d ON d.`pk_id` = vd.`doc_pk_id`",
        "sparse_idf": "LN((%(total_docs)s + 1) / (idf.doc_count + 0.5))",

        # Select documents by pk_id list (SQL-agnostic identifiers must be pre-quoted)
        "select_docs_in_with_tags": """
            SELECT 
                d.{pk_col} AS pk_id,
                d.{id_col} AS id,
                d.{name_col} AS name,
                d.{description_col} AS description,
                d.{timestamp_col} AS timestamp,
                d.{metadata_col} AS metadata,
                (SELECT GROUP_CONCAT(t.{tag_col} SEPARATOR '|')
                   FROM {tags_table} t
                  WHERE t.{tag_doc_pk_col} = d.{pk_col}) AS tags
            FROM {table} d
            WHERE d.{pk_col} IN ({placeholders})
        """,
        
        # Table Options and Feature Flags
        "table_options": "",
        "if_not_exists": "",
    }
    
    # ==========================================
    # Abstract Methods for Database Connection
    # ==========================================
    
    @abstractmethod
    async def execute_sql(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dictionaries."""
        pass
    
    @abstractmethod
    async def execute_sql_no_result(self, sql: str, params: Optional[tuple] = None) -> None:
        """Execute SQL that doesn't return results (INSERT, UPDATE, DELETE)."""
        pass
    
    async def validate_features(self, config: CollectionConfig) -> None:
        """
        Check if database backend supports all features required by the collection configuration.
        
        Args:
            config: Collection configuration to validate against database capabilities
            
        Raises:
            ValueError: If MariaDB doesn't support required features
        """
        # Check if any dense vectors are requested
        if any(VectorType.is_dense(v.type) for v in config.vectors):
            
            # Check if dense vectors are supported
            if not self._db_info.features.get(DatabaseFeatures.DENSE_VECTORS, False):
                raise ValueError(
                    f"Database backend {self._db_info.version} does not support dense vectors. "
                    f"Collection requires dense vector support."
                )    

    # ==========================================
    # Database Probing and Setup
    # ==========================================
    
    async def configure(self) -> None:
        """
        Configure the database with system objects and ensure proper setup.
        """

        # precheck if the database is already configured
        try:
            await self.execute_sql(f"SELECT COUNT(*) FROM {self.quote_identifier(self.meta_collection)}")
            self.logger.info(f"System meta table already exists")
            return
        except Exception as e:
            self.logger.debug(f"Check for system meta table failed: {str(e)}")

        # Create the system meta table if it doesn't exist
        try:
            columns = [
                self.format_sql("column_pk", name="meta_id"),
                self.format_sql("column_varchar", name="collection_name", size=MAX_INTERNAL_COLLECTION_NAME_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_varchar", name="key", size=MAX_METADATA_KEY_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_text", name="value", null_constraint=" NOT NULL"),
                self.format_sql("column_timestamp", name="created_at", default=self.SQL_TEMPLATES["current_timestamp"]),
            ]
            
            # Add index on collection_name for faster lookups (use UUID-scoped name)
            indexes = [
                self.format_sql(
                    "index_unique",
                    name=self._string_to_uuid(f"{self.meta_collection}_collection_name_key"),
                    columns=self.quote_column_list(["collection_name", "key"]),
                )
            ]
            
            meta_sql = self.format_sql("create_table", 
                if_not_exists=" IF NOT EXISTS ",
                table_name=self.meta_collection,
                columns=',\n                '.join(columns),
                indexes=',\n                '.join([''] + indexes) if indexes else '',
                table_options=self.SQL_TEMPLATES.get("table_options", "")
            )
            
            await self.execute_sql_no_result(meta_sql)
            self.logger.info(f"Created system meta table")
            
        except Exception as e:
            if "already exists" in str(e).lower():
                # Another instance beat us to it - that's fine
                self.logger.info(f"System meta table already exists")
            else:
                self.logger.error(f"Failed to create system meta table: {e}")
                raise
        

        # Create the global queue table if it doesn't exist
        try:
            query_columns = [
                self.format_sql("column_varchar", name="queue_id", size=UUID_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_varchar", name="collection_name", size=MAX_INTERNAL_COLLECTION_NAME_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_varchar", name="collection_id", size=UUID_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_varchar", name="doc_id", size=MAX_DOCUMENT_ID_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_varchar", name="status", size=MAX_STATUS_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_longtext", name="document", null_constraint=" NOT NULL"),
                self.format_sql("column_text", name="info", null_constraint=""),
                self.format_sql("column_timestamp", name="created_at", default=self.SQL_TEMPLATES["current_timestamp"]),
                self.format_sql("column_timestamp", name="timestamp", default=self.SQL_TEMPLATES["current_timestamp"]),
                self.format_sql("column_int", name="try_count", null_constraint=" DEFAULT 0"),
            ]
            
            query_indexes = [
                self.format_sql(
                    "index_primary",
                    name=self._string_to_uuid(f"{self.queue_collection}_queue_id"),
                    column="queue_id",
                ),
            ]
            query_simple_indexes = [
                ("collection_doc", self.quote_column_list(["collection_name", "doc_id"])),
                ("timestamp", self.quote_column_list(["timestamp"])),
                ("status", self.quote_column_list(["status"]))
            ]
            
            query_ddl = self.format_sql("create_table", 
                if_not_exists=" IF NOT EXISTS ",
                table_name=self.queue_collection,
                columns=',\n                '.join(query_columns),
                indexes=',\n                '.join([''] + query_indexes) if query_indexes else '',
                table_options=self.SQL_TEMPLATES.get("table_options", "")
            )
            
            await self.execute_sql_no_result(query_ddl)
            # Create simple indexes separately (always use CREATE INDEX)
            for idx_name, idx_columns in query_simple_indexes:
                create_idx_sql = self.format_sql(
                    "create_index_simple",
                    table=self.queue_collection,
                    name=self._string_to_uuid(f"{self.queue_collection}_{idx_name}"),
                    columns=idx_columns,
                )
                try:
                    await self.execute_sql_no_result(create_idx_sql)
                except Exception as e:
                    if "duplicate key name" in str(e).lower():
                        self.logger.info(f"Index {idx_name} already exists")
                    else:
                        self.logger.error(f"Failed to create index {idx_name}: {e}")
                        raise

            self.logger.info(f"Created system queue table")
            
        except Exception as e:
            if "already exists" in str(e).lower():
                # Another instance beat us to it - that's fine
                self.logger.info(f"System queue table already exists")
            else:
                self.logger.error(f"Failed to create system queue table: {e}")
                raise

    async def probe(self) -> None:
        """
        Probe SQL database and store DatabaseInfo internally.
        Tests available features and stores results in _db_info.
        """
        # Test if dense vectors are supported using existing SQL templates
        try:
            # Use existing SQL template for table creation with vector index
            test_table_sql = self.format_sql("vector_test_create")
            await self.execute_sql(test_table_sql)
            
            # Use existing SQL template for table deletion
            drop_sql = self.format_sql("vector_test_drop")
            await self.execute_sql(drop_sql)
            
            dense_vectors_supported = True

        except Exception as e:
            # Feature not supported - store as False
            self.logger.debug(f"Vector support test failed: {str(e)}")
            dense_vectors_supported = False
        
        # Get version using existing SQL template
        try:
            version_result = await self.execute_sql(self.SQL_TEMPLATES["version_query"])
            version = version_result[0].get("version", "unknown") if version_result else "unknown"
        except Exception:
            version = "unknown"
        
        features = {
            DatabaseFeatures.DENSE_VECTORS: dense_vectors_supported
        }
        
        async with self._probe_lock:
            self._db_info_locked = DatabaseInfo(
                version=version,
                features=features
            )
            self._db_info = copy.deepcopy(self._db_info_locked)
    
    # ==========================================
    # SQL Statement Templates (Override for RDBMS-specific syntax)
    # ==========================================
    
    def quote_identifier(self, identifier: str) -> str:
        """
        Quote an identifier according to the database's syntax.
        Escapes quote characters by doubling them (SQL standard).
        Uses the quote_char and quote_identifier templates from SQL_TEMPLATES.
        """
        # Escape quote characters by doubling them (SQL standard)
        escaped_identifier = identifier.replace(self._quote_char, self._quote_char + self._quote_char)
        return self.format_sql("quote_identifier", identifier=escaped_identifier)
        
    def quote_column_list(self, columns: list) -> str:
        """
        Quote a list of column names and join them with commas.
        """
        return ", ".join(self.quote_identifier(col) for col in columns)
    

    
    def generate_batch_insert_sql(self, table_name: str, columns: List[str], batch_size: int) -> str:
        """
        Generate SQL for batch INSERT with multiple VALUES.
        
        Args:
            table_name: Name of the table to insert into
            columns: List of column names
            batch_size: Number of rows to insert in one statement
            
        Returns:
            SQL string for batch insert
        """
        quoted_columns = [self.quote_identifier(col) for col in columns]
        
        # Generate placeholders for one row: (%s, %s, %s)
        row_placeholders = f"({', '.join(['%s'] * len(columns))})"
        
        # Generate multiple rows: (%s, %s, %s), (%s, %s, %s), ...
        all_placeholders = ', '.join([row_placeholders] * batch_size)
        
        return self.format_sql("batch_insert", 
            table=table_name,
            columns=', '.join(quoted_columns),
            placeholders=all_placeholders
        )
    
    def flatten_batch_params(self, data_rows: List[tuple]) -> tuple:
        """
        Flatten a list of data rows into a single tuple for batch insert.
        
        Args:
            data_rows: List of tuples, each representing one row
            
        Returns:
            Flattened tuple of all values
        """
        return tuple(value for row in data_rows for value in row)

    def _get_field_vector_ids(self, collection_config: CollectionConfigInternal) -> Dict[str, int]:
        field_vector_names: List[str] = []
        for vector in collection_config.vectors:
            for field in vector.index_fields:
                field_vector_names.append(f"{field}_{vector.name}")

        field_vector_names.sort()

        return {
            field_vector_name: field_vector_id
            for field_vector_id, field_vector_name in enumerate(field_vector_names)
        }
    
    # ==========================================
    # Helper Methods for Query Management
    # ==========================================
    
    async def insert_query_vectors(
        self,
        collection_name: str,
        rows: List[Tuple[int, int, float, int]],
        table_name: str,
        conn=None
    ) -> None:
        """
        Insert query vector tokens for sparse search with field information.
        
        This method inserts query tokens into the temp query_vectors table. For sparse
        vector types that use IDF, it computes the final weighted query tokens in SQL
        during the insert so the hot search query can stay simple.
        
        Args:
            collection_name: Name of the collection being searched
            rows: Prepared query vector rows as (field_vector_id, token_id, weight, requires_idf)
            conn: Database connection (from transaction context)
        """
        if not rows:
            return

        # Insert in batches
        batch_size = self.DEFAULT_BATCH_SIZE
        docs_table = self.quote_identifier(self.get_table_name(collection_name, self.TableType.DOCUMENTS))
        idf_table = self.quote_identifier(self.get_table_name(collection_name, self.TableType.IDF))
        token_col = self.quote_identifier("token_id")
        doc_count_col = self.quote_identifier("doc_count")
        field_vector_col = self.quote_identifier("field_vector_id")
        weight_col = self.quote_identifier("weight")

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            if not any(row[3] for row in batch):
                sql = self.generate_batch_insert_sql(
                    table_name,
                    ['field_vector_id', 'token_id', 'weight'],
                    len(batch)
                )
                params = self.flatten_batch_params([(fv_id, token_id, weight) for fv_id, token_id, weight, _ in batch])
                await self.execute_sql_no_result(sql, params, conn=conn)
                continue

            select_clauses = []
            params: List[Any] = []
            for field_vector_id, token_id, weight, requires_idf in batch:
                if select_clauses:
                    select_clauses.append(self.format_sql("query_vector_batch_select_next"))
                else:
                    select_clauses.append(self.format_sql("query_vector_batch_select_first"))
                params.extend([field_vector_id, token_id, weight, requires_idf])

            idf_sql = self.format_sql("sparse_idf").replace("%(total_docs)s", "doc_stats.total_docs")
            insert_sql = self.format_sql(
                "query_vector_weighted_insert",
                table=self.quote_identifier(table_name),
                field_vector_col=field_vector_col,
                token_col=token_col,
                weight_col=weight_col,
                idf_expr=idf_sql,
                source_rows=" UNION ALL ".join(select_clauses),
                idf_table=idf_table,
                idf_field_vector_col=field_vector_col,
                idf_token_col=token_col,
                docs_table=docs_table,
                doc_count_col=doc_count_col
            )
            await self.execute_sql_no_result(insert_sql, tuple(params), conn=conn)
    

    
    def format_sql(self, template_name: str, **kwargs) -> str:
        """
        Format a SQL template with the provided parameters.
        
        Args:
            template_name: Name of the template in SQL_TEMPLATES
            **kwargs: Parameters to format the template with
            
        Returns:
            str: The formatted SQL string
        """
        template = self.SQL_TEMPLATES.get(template_name, "")
        
        if not template:
            raise ValueError(f"Missing SQL template: {template_name}")
        
        # Always use format_map with defaultdict to handle missing parameters gracefully
        # This prevents KeyError when template expects parameters we don't provide
        return template.format_map(defaultdict(str, kwargs))
    
    # ==========================================
    # Transaction Management
    # ==========================================
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for transactions that ensures proper commit/rollback.
        This is an abstract method that should be implemented by derived classes.
        """
        raise NotImplementedError("transaction() must be implemented by derived classes")
    
    
    async def execute_in_transaction(self, operations: List[Callable[[], Any]]) -> bool:
        """
        Execute a list of operations within a transaction.
        
        Args:
            operations: List of async functions to execute
            
        Returns:
            bool: True if all operations succeeded, False if any failed
        """
        async with self.transaction() as conn:
            for operation in operations:
                # Execute each operation, passing the connection if needed
                if hasattr(operation, '__code__') and 'conn' in operation.__code__.co_varnames:
                    await operation(conn=conn)
                else:
                    await operation()
            return True

    # ==========================================
    # DatabaseBase Implementation
    # ==========================================
    
    async def create_collection(self, collection_name: str, config: CollectionConfigInternal) -> bool:
        """Create a new collection with specified vector configurations."""
        
        async def insert_metadata():
            # Insert collection config into system meta table
            insert_sql = self.format_sql("insert", 
                table=self.meta_collection, 
                columns=self.quote_column_list(["collection_name", "key", "value"]), 
                placeholders="%s, %s, %s"
            )
            await self.execute_sql_no_result(insert_sql, (collection_name, "config", config.model_dump_json(),))
            await self.execute_sql_no_result(insert_sql, (collection_name, "id", self._string_to_uuid(collection_name),))

        async def create_documents_table():
            # Prepare columns for documents table
            columns = [
                self.format_sql("column_pk", name="pk_id"),
                self.format_sql("column_varchar", name="id", size=MAX_DOCUMENT_ID_LENGTH, null_constraint=" NOT NULL"),
                self.format_sql("column_timestamp", name="timestamp", default=" NOT NULL"),
                self.format_sql("column_text", name="name", null_constraint=""),
                self.format_sql("column_text", name="description", null_constraint=""),
                self.format_sql("column_text", name="metadata", null_constraint=""),
                self.format_sql("column_longtext", name="content", null_constraint="")
            ]
            
            # Add timestamp columns first (needed for indexes)
            columns.extend([
                self.format_sql("column_timestamp", name="created_at", default=self.SQL_TEMPLATES["current_timestamp"]),
                self.format_sql("column_timestamp", name="updated_at", default=self.SQL_TEMPLATES["current_timestamp"])
            ])
            
            # Add dense vector fields
            vector_fields = []
            
            for vector in config.vectors:
                if VectorType.is_dense(vector.type):
                    if not vector.dimensions:
                        raise ValueError(f"Dimensions are required for dense vector {vector.name}")
                    
                    # Create field-specific vectors for each field in index_fields
                    for field in vector.index_fields:
                        field_vector_name = f"{field}_{vector.name}"
                        vector_fields.append(self.format_sql("column_vector", 
                            name=field_vector_name,
                            dimensions=vector.dimensions,
                            null_constraint=" NOT NULL"
                        ))
            
            # Add vector fields to columns
            columns.extend(vector_fields)
            
            # Add indexed metadata columns
            metadata_index_columns = []
            metadata_index_names = []
            if config.metadata_indexes:
                for metadata_index in config.metadata_indexes:
                    column_name = f"meta_{metadata_index.key}"
                    metadata_index_names.append(column_name)
                    
                    if metadata_index.type == MetadataValueType.STRING:
                        metadata_index_columns.append(
                            self.format_sql("column_varchar", name=column_name, size=MAX_METADATA_VALUE_LENGTH, null_constraint="")
                        )
                    elif metadata_index.type in (MetadataValueType.INTEGER, MetadataValueType.FLOAT):
                        metadata_index_columns.append(
                            self.format_sql("column_decimal", name=column_name, null_constraint="")
                        )
                    elif metadata_index.type == MetadataValueType.BOOLEAN:
                        metadata_index_columns.append(
                            self.format_sql("column_int", name=column_name, null_constraint="")
                        )
                    elif metadata_index.type == MetadataValueType.DATETIME:
                        metadata_index_columns.append(
                            self.format_sql("column_timestamp", name=column_name, default="", null_constraint="")
                        )
            
            columns.extend(metadata_index_columns)
            
            # Prepare indexes (after all columns are defined)
            unique_indexes = [
                ("documents_id", self.quote_identifier("id")),
            ]
            documents_simple_indexes = [
                ("documents_timestamp", self.quote_identifier("timestamp")),
                ("documents_created_at", self.quote_identifier("created_at"))
            ]
            
            # Create documents table
            documents_sql = self.format_sql("create_table",
                table_name=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                columns=',\n                '.join(columns),
                indexes='',
                table_options=self.SQL_TEMPLATES.get("table_options", "")
            )
            # self.logger.debug(f"Documents SQL: {documents_sql}")
            await self.execute_sql_no_result(documents_sql)
            # Create unique and simple indexes separately (always use CREATE INDEX)
            for idx_name, idx_columns in unique_indexes:
                table_name = self.get_table_name(collection_name, self.TableType.DOCUMENTS)
                create_unique_sql = self.format_sql(
                    "create_unique_index",
                    table=table_name,
                    name=self._string_to_uuid(f"{collection_name}_{idx_name}"),
                    columns=idx_columns,
                )
                await self.execute_sql_no_result(create_unique_sql)

            # Create simple indexes separately (always use CREATE INDEX)
            for idx_name, idx_columns in documents_simple_indexes:
                table_name = self.get_table_name(collection_name, self.TableType.DOCUMENTS)
                create_idx_sql = self.format_sql(
                    "create_index_simple",
                    table=table_name,
                    name=self._string_to_uuid(f"{collection_name}_{idx_name}"),
                    columns=idx_columns,
                )
                await self.execute_sql_no_result(create_idx_sql)
            
            # Always create vector indexes as separate statements
            if vector_fields:
                for vector in config.vectors:
                    if VectorType.is_dense(vector.type):
                        for field in vector.index_fields:
                            field_vector_name = f"{field}_{vector.name}"
                            
                            # Select the correct template based on distance metric
                            if vector.dense_distance == DenseDistance.COSINE:
                                template_name = "create_index_vector_cosine"
                            elif vector.dense_distance == DenseDistance.DOT:
                                template_name = "create_index_vector_dot"
                            elif vector.dense_distance == DenseDistance.EUCLID:
                                template_name = "create_index_vector_euclid"
                            
                            index_sql = self.format_sql(
                                template_name,
                                table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                                name=self._string_to_uuid(f"{collection_name}_{field_vector_name}"),
                                column=field_vector_name
                            )
                            await self.execute_sql_no_result(index_sql)
            
            # Create indexes on indexed metadata columns
            if metadata_index_names:
                for column_name in metadata_index_names:
                    table_name = self.get_table_name(collection_name, self.TableType.DOCUMENTS)
                    create_idx_sql = self.format_sql(
                        "create_index_simple",
                        table=table_name,
                        name=self._string_to_uuid(f"{collection_name}_{column_name}"),
                        columns=self.quote_column_list([column_name])
                    )
                    await self.execute_sql_no_result(create_idx_sql)
        
        async def create_vector_data_table():
            # Prepare columns
            columns = [
                self.format_sql("column_pk", name="pk_id"),
                self.format_sql("column_bigint", name="doc_pk_id", null_constraint=" NOT NULL"),
                self.format_sql("column_smallint", name="field_vector_id", null_constraint=" NOT NULL"),
                self.format_sql("column_bigint", name="token_id", null_constraint=" NOT NULL"),
                self.format_sql("column_float", name="weight", null_constraint=" NOT NULL")
            ]
            
            # Prepare indexes
            # Unique and FK constraints
            unique_indexes = [
                ("vec_doc_field_token", self.quote_column_list(["field_vector_id", "token_id", "doc_pk_id"]))
            ]
            indexes = [
                self.format_sql(
                    "index_foreign_key",
                    name=self._string_to_uuid(
                        f"{self.get_table_name(collection_name, self.TableType.VECTOR_DATA)}_vd_doc"
                    ),
                    columns=self.quote_identifier("doc_pk_id"),
                    ref_table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                    ref_columns=self.quote_identifier("pk_id"),
                    on_delete=" ON DELETE CASCADE"
                )
            ]
            vector_data_simple_indexes = [
                ("token_vec_doc_idf", self.quote_column_list(["token_id", "field_vector_id", "doc_pk_id"])),
                ("vec_fv_token_doc_weight", self.quote_column_list(["field_vector_id", "token_id", "doc_pk_id", "weight"]))
            ]
            
            # Create vector data table
            vector_data_sql = self.format_sql("create_table",
                table_name=self.get_table_name(collection_name, self.TableType.VECTOR_DATA),
                columns=',\n                '.join(columns),
                indexes=',\n                '.join([''] + indexes) if indexes else '',
                table_options=self.SQL_TEMPLATES.get("table_options", "")
            )
            await self.execute_sql_no_result(vector_data_sql)

            # Create unique and simple indexes separately (always use CREATE INDEX)
            table_name = self.get_table_name(collection_name, self.TableType.VECTOR_DATA)
            for idx_name, idx_columns in unique_indexes:
                create_unique_sql = self.format_sql(
                    "create_unique_index",
                    table=table_name,
                    name=self._string_to_uuid(f"{collection_name}_{idx_name}"),
                    columns=idx_columns,
                )
                await self.execute_sql_no_result(create_unique_sql)
            # Create simple indexes separately (always use CREATE INDEX)
            for idx_name, idx_columns in vector_data_simple_indexes:
                table_name = self.get_table_name(collection_name, self.TableType.VECTOR_DATA)
                create_idx_sql = self.format_sql(
                    "create_index_simple",
                    table=table_name,
                    name=self._string_to_uuid(f"{collection_name}_{idx_name}"),
                    columns=idx_columns,
                )
                await self.execute_sql_no_result(create_idx_sql)

        async def create_tags_table():
            # Prepare columns for tags table: doc_pk_id + tag
            tag_columns = [
                self.format_sql("column_bigint", name="doc_pk_id", null_constraint=" NOT NULL"),
                self.format_sql("column_varchar", name="tag", size=MAX_DOCUMENT_TAG_LENGTH, null_constraint=" NOT NULL"),
            ]

            # Primary key on (doc_pk_id, tag)
            tag_indexes = [
                self.format_sql(
                    "index_primary_multi",
                    name=self._string_to_uuid(
                        f"{self.get_table_name(collection_name, self.TableType.TAGS)}"
                    ),
                    columns=self.quote_column_list(["doc_pk_id", "tag"])
                ),
                self.format_sql(
                    "index_foreign_key",
                    name=self._string_to_uuid(
                        f"{self.get_table_name(collection_name, self.TableType.TAGS)}_tags_doc"
                    ),
                    columns=self.quote_identifier("doc_pk_id"),
                    ref_table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                    ref_columns=self.quote_identifier("pk_id"),
                    on_delete=" ON DELETE CASCADE"
                )
            ]

            tags_sql = self.format_sql(
                "create_table",
                table_name=self.get_table_name(collection_name, self.TableType.TAGS),
                columns=',\n                '.join(tag_columns),
                indexes=',\n                '.join([''] + tag_indexes) if tag_indexes else '',
                table_options=self.SQL_TEMPLATES.get("table_options", "")
            )
            await self.execute_sql_no_result(tags_sql)

        async def create_idf_table():
            # Prepare columns for IDF table: field_vector_id, token_id, doc_count
            idf_columns = [
                self.format_sql("column_smallint", name="field_vector_id", null_constraint=" NOT NULL"),
                self.format_sql("column_bigint", name="token_id", null_constraint=" NOT NULL"),
                self.format_sql("column_int", name="doc_count", null_constraint=" NOT NULL")
            ]

            # Primary key on (field_vector_id, token_id)
            idf_indexes = [
                self.format_sql(
                    "index_primary_multi",
                    name=self._string_to_uuid(
                        f"{self.get_table_name(collection_name, self.TableType.IDF)}"
                    ),
                    columns=self.quote_column_list(["field_vector_id", "token_id"]),
                )
            ]

            idf_sql = self.format_sql(
                "create_table",
                table_name=self.get_table_name(collection_name, self.TableType.IDF),
                columns=',\n                '.join(idf_columns),
                indexes=',\n                '.join([''] + idf_indexes) if idf_indexes else '',
                table_options=self.SQL_TEMPLATES.get("table_options", "")
            )
            await self.execute_sql_no_result(idf_sql)
        
        # Execute all creation operations in a transaction
        operations = [
            insert_metadata,
            create_documents_table,
            create_vector_data_table,
            create_tags_table,
            create_idf_table
        ]
        await self.execute_in_transaction(operations)
        return True
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        # Use our TableType enum to construct the expected table name pattern
        # For a collection named 'my_collection', we expect a table named 'my_collection_docs'
        docs_suffix = self.TableType.DOCUMENTS.value
        
        # Query information_schema to find tables that match our naming pattern
        # We look for tables that end with '_docs' and extract the collection name
        sql = self.format_sql("list_collections_query",
            docs_suffix_length=len(docs_suffix) + 1,
            docs_suffix=docs_suffix,
            database=self.database,
            prefix=APP_PREFIX
        )
        results = await self.execute_sql(sql)
        
        # Filter out any empty collection names and return the list
        collection_names = [row['collection_name'] for row in results if row['collection_name']]
        return collection_names
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its data."""
        
        async with self.transaction() as conn:
            # Drop collection tables in one statement (in FK-safe order)
            tables = f"{self.quote_identifier(self.get_table_name(collection_name, self.TableType.VECTOR_DATA))}, {self.quote_identifier(self.get_table_name(collection_name, self.TableType.TAGS))}, {self.quote_identifier(self.get_table_name(collection_name, self.TableType.IDF))}, {self.quote_identifier(self.get_table_name(collection_name, self.TableType.DOCUMENTS))}"
            drop_sql = self.format_sql("drop_table", tables=tables)
            await self.execute_sql_no_result(drop_sql, conn=conn)
            
            # Delete collection config from system meta table
            delete_sql = self.format_sql("delete", 
                table=self.meta_collection,
                where_clause=f'{self.quote_identifier("collection_name")}=%s'
            )
            await self.execute_sql_no_result(delete_sql, (collection_name,), conn=conn)
            
            # Clean up queue entries for this collection
            await self.delete_from_queue_by_collection(collection_name)
            
            return True
    
    async def empty_collection(self, collection_name: str) -> bool:
        """Remove all documents by dropping and recreating the collection using stored config."""
        # Load stored internal config from meta
        meta_result = await self.execute_sql(
            self.format_sql(
                "select",
                table=self.meta_collection,
                columns=self.quote_identifier("value"),
                where=f" WHERE {self.quote_identifier('collection_name')}=%s AND {self.quote_identifier('key')}='config'"
            ),
            (collection_name,)
        )
        if not meta_result:
            raise Exception("Configuration not found")
            
        config_data = meta_result[0]["value"]
        config_dict = json.loads(config_data) if isinstance(config_data, str) else config_data
        internal_config = CollectionConfigInternal.model_validate(config_dict)

        # Drop and recreate the collection
        await self.delete_collection(collection_name)
        await self.create_collection(collection_name, internal_config)
        return True
    
    async def add_documents(self, collection_name: str, documents_with_vectors: List[DocumentWithVectors], is_new: bool, store_content: bool, collection_config: CollectionConfigInternal, lock_client: LockClient) -> None:
        """Add documents with their vectors to the collection."""
        lock_name = f"collection-ingest-{self._string_to_uuid(collection_name)}"
        async with lock_client.acquire(lock_name, timeout=COLLECTION_INGEST_LOCK_TIMEOUT):
            await self._add_documents_impl(collection_name, documents_with_vectors, is_new, store_content, collection_config)

    async def _add_documents_impl(self, collection_name: str, documents_with_vectors: List[DocumentWithVectors], is_new: bool, store_content: bool, collection_config: CollectionConfigInternal) -> None:
        metadata_indexes = collection_config.metadata_indexes or []
        field_vector_ids = self._get_field_vector_ids(collection_config)
        
        async def insert_document(document_with_vectors: DocumentWithVectors, conn=None):
            # Prepare metadata JSON
            if document_with_vectors.metadata:
                metadata_dict = {k: v.model_dump() for k, v in document_with_vectors.metadata.items()}
                metadata_json = json.dumps(metadata_dict)
            else:
                metadata_json = None
            
            # Build columns and values for dense vectors
            columns = ["id", "timestamp", "name", "description", "metadata"]
            values = [
                document_with_vectors.id,
                document_with_vectors.timestamp,
                document_with_vectors.name,
                document_with_vectors.description,
                metadata_json
            ]
            
            # Add indexed metadata columns and values
            if metadata_indexes and document_with_vectors.metadata:
                for metadata_index in metadata_indexes:
                    column_name = f"meta_{metadata_index.key}"
                    columns.append(column_name)
                    
                    # Extract value from metadata if present
                    if metadata_index.key in document_with_vectors.metadata:
                        meta_value = document_with_vectors.metadata[metadata_index.key]
                        if metadata_index.type == MetadataValueType.BOOLEAN:
                            values.append(1 if meta_value.value else 0)
                        elif metadata_index.type == MetadataValueType.DATETIME:
                            values.append(datetime.fromisoformat(meta_value.value.replace('Z', '+00:00')))
                        else:
                            values.append(meta_value.value)
                    else:
                        values.append(None)
            
            # Add content column and value if store_content is enabled
            if store_content:
                columns.append("content")
                values.append(document_with_vectors.content)
            else:
                columns.append("content")
                values.append(None)
            
            # Build placeholders dynamically - regular columns use %s, dense vectors use Vec_FromText(%s)
            placeholders = ["%s"] * len(columns)
            
            # Add dense vector columns and values
            for vector in document_with_vectors.vectors:
                if VectorType.is_dense(vector.vector_type) and vector.dense_vector:
                    # VectorData.field is the specific field this vector is for
                    field_vector_name = f"{vector.field}_{vector.vector_name}"
                    columns.append(field_vector_name)
                    # Convert Python list to MariaDB vector format
                    # Round to 6 decimal places for MariaDB compatibility (32-bit float precision)
                    dense_vector_str = f"[{','.join(str(val) for val in vector.dense_vector)}]"
                    # Use the template but replace the placeholder with %s for parameter binding
                    placeholder = self.format_sql("dense_vector_insert", vector_values="%s")
                    placeholders.append(placeholder)
                    # Add just the vector array string as the value
                    values.append(dense_vector_str)
            
            # Use the insert template with get_last_id placeholder
            insert_sql = self.format_sql("insert",
                table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                columns=self.quote_column_list(columns),
                placeholders=", ".join(placeholders),
                get_last_id=" " + self.SQL_TEMPLATES["last_insert_id"]
            )
            
            result = await self.execute_sql(
                insert_sql,
                tuple(values),
                skip_first=True,
                conn=conn
            )
            
            # The result will contain the inserted pk_id
            return result[0]["pk_id"] if result else None

        async def update_document(document_with_vectors: DocumentWithVectors, conn=None):
            # Prepare metadata JSON
            if document_with_vectors.metadata:
                metadata_dict = {k: v.model_dump() for k, v in document_with_vectors.metadata.items()}
                metadata_json = json.dumps(metadata_dict)
            else:
                metadata_json = None
            
            # Build columns and values for dense vectors
            set_columns = ["timestamp", "name", "description", "metadata"]
            set_values = [
                document_with_vectors.timestamp,
                document_with_vectors.name,
                document_with_vectors.description,
                metadata_json
            ]
            
            # Add indexed metadata columns and values
            if metadata_indexes and document_with_vectors.metadata:
                for metadata_index in metadata_indexes:
                    column_name = f"meta_{metadata_index.key}"
                    set_columns.append(column_name)
                    
                    # Extract value from metadata if present
                    if metadata_index.key in document_with_vectors.metadata:
                        meta_value = document_with_vectors.metadata[metadata_index.key]
                        if metadata_index.type == MetadataValueType.BOOLEAN:
                            set_values.append(1 if meta_value.value else 0)
                        elif metadata_index.type == MetadataValueType.DATETIME:
                            set_values.append(datetime.fromisoformat(meta_value.value.replace('Z', '+00:00')))
                        else:
                            set_values.append(meta_value.value)
                    else:
                        set_values.append(None)
            
            # Add content column and value if store_content is enabled
            if store_content:
                set_columns.append("content")
                set_values.append(document_with_vectors.content)
            else:
                set_columns.append("content")
                set_values.append(None)
            
            # Build set clause dynamically - regular columns use %s, dense vectors use Vec_FromText(%s)
            set_clause_parts = []
            for col in set_columns:
                set_clause_parts.append(f'{self.quote_identifier(col)}=%s')
            
            # Add dense vector columns and values
            for vector in document_with_vectors.vectors:
                if VectorType.is_dense(vector.vector_type) and vector.dense_vector:
                    # VectorData.field is the specific field this vector is for
                    field_vector_name = f"{vector.field}_{vector.vector_name}"
                    set_columns.append(field_vector_name)
                    # Convert Python list to MariaDB vector format
                    # Round to 6 decimal places for MariaDB compatibility (32-bit float precision)
                    dense_vector_str = f"[{','.join(str(val) for val in vector.dense_vector)}]"
                    # Use the template but replace the placeholder with %s for parameter binding
                    set_clause_parts.append(f'{self.quote_identifier(field_vector_name)}={self.format_sql("dense_vector_insert", vector_values="%s")}')
                    # Add just the vector array string as the value
                    set_values.append(dense_vector_str)
            
            # Update document using dynamically built set clause
            set_clause = ", ".join(set_clause_parts)
            
            update_sql = self.format_sql("update", 
                table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                set_clause=set_clause,
                where_clause=f'{self.quote_identifier("id")}=%s'
            )
            
            # Add document ID to the end for the WHERE clause
            set_values.append(document_with_vectors.id)
            
            await self.execute_sql_no_result(
                update_sql,
                tuple(set_values),
                conn=conn
            )
            
            # Get the pk_id of the updated document (need to query since UPDATE doesn't set LAST_INSERT_ID)
            select_sql = self.format_sql("select",
                table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                columns=self.quote_identifier("pk_id"),
                where=f" WHERE {self.quote_identifier('id')}=%s"
            )
            
            result = await self.execute_sql(select_sql, (document_with_vectors.id,), conn=conn)
            
            return result[0]["pk_id"] if result else None

        async def delete_existing_vectors(doc_pk_id: int, conn=None):
            # Remove prior vector rows for this document before inserting new ones
            delete_sql = self.format_sql("delete", 
                table=self.get_table_name(collection_name, self.TableType.VECTOR_DATA),
                where_clause=f'{self.quote_identifier("doc_pk_id")}=%s'
            )
            await self.execute_sql_no_result(delete_sql, (doc_pk_id,), conn=conn)
            
            # Update IDF counts by decrementing doc_count for existing tokens using a single JOIN query
            # This is much more efficient than querying first, then batching updates
            idf_update_sql = self.format_sql("update_join",
                table=self.get_table_name(collection_name, self.TableType.IDF),
                alias="idf",
                subquery=f"SELECT DISTINCT {self.quote_identifier('field_vector_id')}, {self.quote_identifier('token_id')} FROM {self.quote_identifier(self.get_table_name(collection_name, self.TableType.VECTOR_DATA))} WHERE {self.quote_identifier('doc_pk_id')} = %s",
                subquery_alias="vd",
                join_conditions="vd.field_vector_id = idf.field_vector_id AND vd.token_id = idf.token_id",
                set_clause=f"{self.quote_identifier('doc_count')} = {self.quote_identifier('doc_count')} - 1"
            )
            
            # Execute the single IDF update query
            await self.execute_sql_no_result(idf_update_sql, (doc_pk_id,), conn=conn)
            
            # Now clean up any IDF records that have doc_count = 0 after the decrement
            idf_cleanup_sql = self.format_sql("delete_join",
                table=self.get_table_name(collection_name, self.TableType.IDF),
                alias="idf",
                subquery=f"SELECT DISTINCT {self.quote_identifier('field_vector_id')}, {self.quote_identifier('token_id')} FROM {self.quote_identifier(self.get_table_name(collection_name, self.TableType.VECTOR_DATA))} WHERE {self.quote_identifier('doc_pk_id')} = %s",
                subquery_alias="vd",
                join_conditions="vd.field_vector_id = idf.field_vector_id AND vd.token_id = idf.token_id",
                where_clause=f"{self.quote_identifier('doc_count')} = 0"
            )
            
            # Execute the IDF cleanup query
            await self.execute_sql_no_result(idf_cleanup_sql, (doc_pk_id,), conn=conn)
        
        async def insert_vector_data(document_with_vectors: DocumentWithVectors, doc_pk_id: int, conn=None):
            # Insert sparse vector data if present
            if document_with_vectors.vectors:
                # Build rows for all sparse vectors from provided VectorData
                # AND collect unique token occurrences for IDF updates in the same loop
                rows: List[tuple] = []
                token_occurrences = set()
                
                for v in document_with_vectors.vectors:
                    if v.sparse_indices and v.sparse_values:
                        field_vector_name = f"{v.field}_{v.vector_name}"
                        field_vector_id = field_vector_ids[field_vector_name]
                        for token_id, weight in zip(v.sparse_indices, v.sparse_values):
                            rows.append((doc_pk_id, field_vector_id, token_id, weight))
                            # Collect unique tokens for IDF updates
                            token_occurrences.add((field_vector_id, token_id))

                if rows:
                    batch_size = self.DEFAULT_BATCH_SIZE
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i + batch_size]
                        # Always use batch insert - it handles any size efficiently
                        sql = self.generate_batch_insert_sql(
                            self.get_table_name(collection_name, self.TableType.VECTOR_DATA),
                            ['doc_pk_id', 'field_vector_id', 'token_id', 'weight'],
                            len(batch)
                        )
                        params = self.flatten_batch_params(batch)
                        await self.execute_sql_no_result(sql, params, conn=conn)

                    # Now upsert IDF records for all unique (field_vector_id, token_id) pairs
                    if token_occurrences:
                        # Convert to list and sort deterministically to impose a stable lock order
                        idf_rows = sorted((field_vector_id, token_id, 1) for field_vector_id, token_id in token_occurrences)
                        
                        # Process IDF updates in batches
                        for i in range(0, len(idf_rows), batch_size):
                            batch = idf_rows[i:i + batch_size]
                            
                            # Build the upsert SQL using the new upsert template
                            columns = ['field_vector_id', 'token_id', 'doc_count']
                            values_placeholders = ', '.join(['(%s, %s, %s)'] * len(batch))
                            update_clause = self.SQL_TEMPLATES["upsert_update_doc_count"].format(
                                table=self.get_table_name(collection_name, self.TableType.IDF)
                            )
                            
                            upsert_sql = self.format_sql(
                                "upsert",
                                table=self.get_table_name(collection_name, self.TableType.IDF),
                                columns=', '.join([self.quote_identifier(col) for col in columns]),
                                values=values_placeholders,
                                update_clause=update_clause,
                                conflict_columns=self.quote_column_list(["field_vector_id", "token_id"]),
                            )
                            
                            # Flatten the batch parameters
                            params = []
                            for field_vector_id, token_id, doc_count in batch:
                                params.extend([field_vector_id, token_id, doc_count])
                            
                            await self.execute_sql_no_result(upsert_sql, tuple(params), conn=conn)

        async def delete_existing_tags(doc_pk_id: int, conn=None):
            delete_sql = self.format_sql(
                "delete",
                table=self.get_table_name(collection_name, self.TableType.TAGS),
                where_clause=f'{self.quote_identifier("doc_pk_id")}=%s'
            )
            await self.execute_sql_no_result(delete_sql, (doc_pk_id,), conn=conn)

        async def insert_tags(document_with_vectors: DocumentWithVectors, doc_pk_id: int, conn=None):
            tags = document_with_vectors.tags
            if not tags:
                return
            rows: List[tuple] = [(doc_pk_id, tag) for tag in tags]
            batch_size = self.DEFAULT_BATCH_SIZE
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                sql = self.generate_batch_insert_sql(
                    self.get_table_name(collection_name, self.TableType.TAGS),
                    ['doc_pk_id', 'tag'],
                    len(batch)
                )
                params = self.flatten_batch_params(batch)
                await self.execute_sql_no_result(sql, params, conn=conn)
        
        # Use a single transaction for all documents
        async with self.transaction() as conn:
            for document_with_vectors in documents_with_vectors:
                # Execute document insertion/update and get pk_id
                if is_new:
                    doc_pk_id = await insert_document(document_with_vectors, conn)
                else:
                    doc_pk_id = await update_document(document_with_vectors, conn)
                
                if doc_pk_id is None:
                    raise RuntimeError(f"Failed to get pk_id for document {document_with_vectors.id}")
                
                # Execute vector operations with the pk_id
                if is_new:
                    # For new documents, just insert vectors (no existing ones to delete)
                    await insert_vector_data(document_with_vectors, doc_pk_id, conn)
                    await insert_tags(document_with_vectors, doc_pk_id, conn)
                else:
                    # For existing documents, delete old vectors then insert new ones
                    await delete_existing_vectors(doc_pk_id, conn)
                    await insert_vector_data(document_with_vectors, doc_pk_id, conn)
                    await delete_existing_tags(doc_pk_id, conn)
                    await insert_tags(document_with_vectors, doc_pk_id, conn)
    
    async def get_documents(self, collection_name: str, document_ids: List[str], suppress_not_found: bool = False) -> List[Optional[DocumentWithVectors]]:
        """
        Retrieve multiple documents by IDs.
        
        Args:
            collection_name: Name of the collection to retrieve from
            document_ids: List of document IDs to retrieve
            suppress_not_found: If True, don't raise AmgixNotFound when documents are missing (default: False)
            
        Returns:
            List[Optional[DocumentWithVectors]]: List of documents in the same order as document_ids, None for missing documents
            
        Raises:
            AmgixNotFound: If suppress_not_found is False and not all documents are found
        """
        
        try:
            # Get all documents in one query
            placeholders = ', '.join(['%s'] * len(document_ids))
            columns = ["pk_id", "id", "timestamp", "name", "description", "metadata", "content"]
            sql = self.format_sql("select", 
                table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                columns=self.quote_column_list(columns),
                where=f" WHERE {self.quote_identifier('id')} IN ({placeholders})"
            )
            result = await self.execute_sql(sql, tuple(document_ids))
            
            # Check if we got the expected number of results
            if len(result) != len(document_ids) and not suppress_not_found:
                found_ids = {row["id"] for row in result}
                missing_ids = set(document_ids) - found_ids
                raise AmgixNotFound(f"Documents not found for document_ids: {', '.join(missing_ids)}")
            
            # Create a map of doc_id to row data
            doc_rows = {row["id"]: row for row in result}
            doc_pk_ids = {row["id"]: row["pk_id"] for row in result}
            
            # Get all tags in one query
            if doc_pk_ids:
                pk_id_placeholders = ', '.join(['%s'] * len(doc_pk_ids))
                tags_sql = self.format_sql("select",
                    table=self.get_table_name(collection_name, self.TableType.TAGS),
                    columns=self.quote_column_list(["doc_pk_id", "tag"]),
                    where=f" WHERE {self.quote_identifier('doc_pk_id')} IN ({pk_id_placeholders})"
                )
                tags_result = await self.execute_sql(tags_sql, tuple(doc_pk_ids.values()))
                
                # Group tags by doc_pk_id
                tags_by_pk_id = {}
                for tag_row in tags_result:
                    pk_id = tag_row["doc_pk_id"]
                    if pk_id not in tags_by_pk_id:
                        tags_by_pk_id[pk_id] = []
                    tags_by_pk_id[pk_id].append(tag_row["tag"])
            else:
                tags_by_pk_id = {}
            
            # Build documents in the same order as document_ids
            documents = []
            for doc_id in document_ids:
                if doc_id not in doc_rows:
                    documents.append(None)
                    continue
                row = doc_rows[doc_id]
                pk_id = doc_pk_ids[doc_id]
                tags = tags_by_pk_id.get(pk_id, [])
                
                # Parse metadata JSON from TEXT column
                metadata = None
                metadata_raw = row.get("metadata")
                if metadata_raw:
                    metadata = json.loads(metadata_raw)
                
                doc_data = {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "name": row.get("name"),
                    "description": row.get("description"),
                    "content": row.get("content"),
                    "metadata": metadata,
                    "vectors": []
                }
                
                if tags:
                    doc_data["tags"] = tags
                
                documents.append(DocumentWithVectors.from_dict(doc_data, store_content=True, skip_validation=True))
            
            return documents
            
        except Exception:
            raise
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            # Use transaction context manager for atomic operations
            async with self.transaction() as conn:
                # First, get the document's pk_id to update IDF counts
                select_sql = self.format_sql("select",
                    table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                    columns=self.quote_column_list(["pk_id"]),
                    where=f" WHERE {self.quote_identifier('id')}=%s"
                )

                result = await self.execute_sql(select_sql, (document_id,), conn=conn)
                
                if not result:
                    # Document doesn't exist, throw an exception
                    raise AmgixNotFound(f"Document with ID '{document_id}' not found in collection '{get_user_collection_name(collection_name)}'")
                
                doc_pk_id = result[0]["pk_id"]
                
                # Update IDF counts by decrementing doc_count for all tokens in this document
                # This must happen BEFORE deleting the document (due to CASCADE)
                idf_update_sql = self.format_sql("update_join",
                    table=self.get_table_name(collection_name, self.TableType.IDF),
                    alias="idf",
                    subquery=f"SELECT DISTINCT {self.quote_identifier('field_vector_id')}, {self.quote_identifier('token_id')} FROM {self.quote_identifier(self.get_table_name(collection_name, self.TableType.VECTOR_DATA))} WHERE {self.quote_identifier('doc_pk_id')} = %s",
                    subquery_alias="vd",
                    join_conditions="vd.field_vector_id = idf.field_vector_id AND vd.token_id = idf.token_id",
                    set_clause=f"{self.quote_identifier('doc_count')} = {self.quote_identifier('doc_count')} - 1"
                )
                
                # Execute the IDF update query
                await self.execute_sql_no_result(idf_update_sql, (doc_pk_id,), conn=conn)
                
                # Now clean up any IDF records that have doc_count = 0 after the decrement
                idf_cleanup_sql = self.format_sql("delete_join",
                    table=self.get_table_name(collection_name, self.TableType.IDF),
                    alias="idf",
                    subquery=f"SELECT DISTINCT {self.quote_identifier('field_vector_id')}, {self.quote_identifier('token_id')} FROM {self.quote_identifier(self.get_table_name(collection_name, self.TableType.VECTOR_DATA))} WHERE {self.quote_identifier('doc_pk_id')} = %s",
                    subquery_alias="vd",
                    join_conditions="vd.field_vector_id = idf.field_vector_id AND vd.token_id = idf.token_id",
                    where_clause=f"{self.quote_identifier('doc_count')} = 0"
                )
                
                # Execute the IDF cleanup query
                await self.execute_sql_no_result(idf_cleanup_sql, (doc_pk_id,), conn=conn)
                
                # Now delete the document (CASCADE will handle vector_data and tags)
                delete_sql = self.format_sql("delete", 
                    table=self.get_table_name(collection_name, self.TableType.DOCUMENTS),
                    where_clause=f'{self.quote_identifier("id")}=%s'
                )
                await self.execute_sql_no_result(delete_sql, (document_id,), conn=conn)
                
                return True
                
        except Exception:
            raise
    
    async def search(self, collection_name: str, query: SearchQueryWithVectors, collection_config: CollectionConfigInternal) -> List[SearchResult]:
        """
        Perform a hybrid search on the collection using precalculated vectors.
        """
        field_vector_ids = self._get_field_vector_ids(collection_config)
        vector_config_map = {vc.name: vc for vc in collection_config.vectors}
        weight_lookup = {(x.vector_name, x.field): x.weight for x in query.vector_weights}
        search_arms = []

        for vector_data in query.vectors:
            field_vector_name = f"{vector_data.field}_{vector_data.vector_name}"
            weight = weight_lookup.get((vector_data.vector_name, vector_data.field), 1.0)
            if weight == 0:
                continue

            field_vector_id = field_vector_ids[field_vector_name]
            sparse_tokens = None
            requires_idf = 0
            if not VectorType.is_dense(vector_data.vector_type):
                vector_config = vector_config_map.get(vector_data.vector_name)
                requires_idf = 1 if (
                    vector_data.vector_type in VectorType.custom_tokenization() or (
                        vector_config is not None
                        and vector_config.model is not None
                        and vector_config.model.lower() == "qdrant/bm25"
                    )
                ) else 0
                sparse_tokens = list(zip(vector_data.sparse_indices, vector_data.sparse_values))

            search_arms.append((vector_data, field_vector_id, weight, sparse_tokens, requires_idf))

        return await self._perform_search(
            collection_name,
            query,
            search_arms,
            collection_config,
            field_vector_ids
        )

    def _normalize_metadata_filter_value(
        self,
        key: str,
        value: Any,
        metadata_indexes_map: Dict[str, str]
    ) -> Any:
        expected_type = metadata_indexes_map.get(key)
        if expected_type == MetadataValueType.BOOLEAN and isinstance(value, bool):
            return 1 if value else 0
        if expected_type == MetadataValueType.DATETIME and isinstance(value, str):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value

    def _convert_metadata_filter_to_sql(
        self,
        metadata_filter: MetadataFilter,
        collection_config: CollectionConfigInternal,
    ) -> Tuple[str, Dict[str, Any]]:
        """Convert MetadataFilter to a SQL WHERE clause fragment and a params dict.

        When a node has multiple boolean operators (and_, or_, not_) at the same
        level, their resulting SQL parts are combined with AND. For example,
        {and: [A, B], or: [C, D]} produces (A AND B) AND (C OR D).
        """
        if not metadata_filter:
            return "", {}

        filter_params: Dict[str, Any] = {}
        metadata_indexes_map = {idx.key: idx.type for idx in (collection_config.metadata_indexes or [])}

        def convert_node(node: MetadataFilter, counter: int) -> Tuple[str, int]:
            if node.key:
                column_name = f"d.{self.quote_identifier(f'meta_{node.key}')}"
                param_name = f"metadata_filter_{counter}"
                filter_params[param_name] = self._normalize_metadata_filter_value(
                    node.key,
                    node.value,
                    metadata_indexes_map
                )

                if node.op == "eq":
                    sql_op = "="
                elif node.op == "gt":
                    sql_op = ">"
                elif node.op == "gte":
                    sql_op = ">="
                elif node.op == "lt":
                    sql_op = "<"
                elif node.op == "lte":
                    sql_op = "<="
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {node.op}")

                return f"{column_name} {sql_op} %({param_name})s", counter + 1

            sql_parts: List[str] = []

            if node.and_:
                and_parts: List[str] = []
                for child in node.and_:
                    child_sql, counter = convert_node(child, counter)
                    if child_sql:
                        and_parts.append(f"({child_sql})")
                if and_parts:
                    sql_parts.append(f"({' AND '.join(and_parts)})")

            if node.or_:
                or_parts: List[str] = []
                for child in node.or_:
                    child_sql, counter = convert_node(child, counter)
                    if child_sql:
                        or_parts.append(f"({child_sql})")
                if or_parts:
                    sql_parts.append(f"({' OR '.join(or_parts)})")

            if node.not_:
                child_sql, counter = convert_node(node.not_, counter)
                if child_sql:
                    sql_parts.append(f"NOT ({child_sql})")

            return " AND ".join(sql_parts), counter

        sql_fragment, _ = convert_node(metadata_filter, 0)
        return sql_fragment, filter_params

    
    async def _perform_search(
        self, 
        collection_name: str, 
        query: SearchQueryWithVectors,
        search_arms: List[Tuple[Any, int, float, Optional[List[Tuple[int, float]]], int]],
        collection_config: CollectionConfigInternal,
        field_vector_ids: Dict[str, int],
    ) -> List[SearchResult]:
        """
        Internal search method that executes vector arms in parallel and fuses the results.
        """
        if not search_arms:
            raise AmgixValidationError("Search query has no active vectors to search with")
        
        docs_table = self.get_table_name(collection_name, self.TableType.DOCUMENTS)
        vector_table = self.get_table_name(collection_name, self.TableType.VECTOR_DATA)
        query_vectors_table = self.get_table_name("", self.TableType.QUERY_VECTORS)
        tags_table = self.get_table_name(collection_name, self.TableType.TAGS)

        has_document_tags_filter = bool(query.document_tags)
        metadata_filter_sql = ""
        filter_params: Dict[str, Any] = {}
        if query.metadata_filter:
            metadata_filter_sql, filter_params = self._convert_metadata_filter_to_sql(
                query.metadata_filter,
                collection_config,
            )
        has_document_filter = has_document_tags_filter or bool(metadata_filter_sql)

        tag_filter_condition = ""
        base_params: Dict[str, Any] = dict(filter_params)
        if has_document_tags_filter:
            base_params["document_tags"] = tuple(query.document_tags)
            if query.document_tags_match_all:
                base_params["document_tags_count"] = len(query.document_tags)
                tag_filter_template = self.format_sql("tags_filter_and", tags_table=tags_table).strip()
            else:
                tag_filter_template = self.format_sql("tags_filter", tags_table=tags_table).strip()
            if tag_filter_template.upper().startswith("WHERE "):
                tag_filter_condition = tag_filter_template[6:]
            else:
                tag_filter_condition = tag_filter_template

        filter_conditions: List[str] = []
        if tag_filter_condition:
            filter_conditions.append(f"({tag_filter_condition})")
        if metadata_filter_sql:
            filter_conditions.append(f"({metadata_filter_sql})")

        tags_filter = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ""
        sparse_documents_join = (
            self.format_sql("sparse_documents_join", documents_table=docs_table)
            if has_document_filter
            else ""
        )
        vector_config_map = {vc.name: vc for vc in collection_config.vectors}
        field_vector_names_by_id = (
            {field_vector_id: field_vector_name for field_vector_name, field_vector_id in field_vector_ids.items()}
            if query.raw_scores else None
        )
        raw_scores_map: Dict[int, List[VectorScore]] = {}
        prefetch_limit = int(query.limit * SEARCH_PREFETCH_MULTIPLIER)
        temp_cols = [
            self.format_sql("column_smallint", name="field_vector_id", null_constraint=" NOT NULL"),
            self.format_sql("column_bigint", name="token_id", null_constraint=" NOT NULL"),
            self.format_sql("column_float", name="weight", null_constraint=" NOT NULL"),
            self.format_sql("index_primary_multi", name=query_vectors_table, columns=self.quote_column_list(["field_vector_id", "token_id"]))
        ]
        create_temp_sql = self.format_sql(
            "create_temp_table",
            table_name=query_vectors_table,
            columns=',\n                '.join(temp_cols),
            table_options=self.SQL_TEMPLATES.get("table_options", "")
        )
        async def execute_arm(search_arm):
            vector_data, field_vector_id, _, sparse_tokens, requires_idf = search_arm
            arm_rows = []
            params = dict(base_params)
            params["prefetch_limit"] = prefetch_limit

            async with self.transaction() as conn:
                if sparse_tokens is None:
                    field_vector_name = f"{vector_data.field}_{vector_data.vector_name}"
                    vector_config = vector_config_map.get(vector_data.vector_name)
                    distance_expr = self.format_sql(
                        f"dense_distance_{vector_config.dense_distance}",
                        field_name=field_vector_name,
                        param_name="dense_vector"
                    )
                    sql = self.format_sql(
                        "dense_union_single",
                        field_vector_id=field_vector_id,
                        documents_table=docs_table,
                        distance_expr=distance_expr,
                        prefetch_limit=prefetch_limit,
                        tags_filter=tags_filter
                    )
                    params["dense_vector"] = f"[{','.join(str(val) for val in vector_data.dense_vector)}]"
                    ef_sql = self.format_sql("ef_search", ef_search=prefetch_limit)
                    if ef_sql:
                        await self.execute_sql_no_result(ef_sql, conn=conn)
                else:
                    await self.execute_sql_no_result(create_temp_sql, conn=conn)
                    rows = [
                        (field_vector_id, token_id, weight, requires_idf)
                        for token_id, weight in sparse_tokens
                    ]
                    await self.insert_query_vectors(
                        collection_name,
                        rows,
                        table_name=query_vectors_table,
                        conn=conn
                    )
                    sql = self.format_sql(
                        "sparse_union_single",
                        vector_data_table=vector_table,
                        query_vectors_table=query_vectors_table,
                        field_vector_id=field_vector_id,
                        prefetch_limit=prefetch_limit,
                        sparse_documents_join=sparse_documents_join,
                        tags_filter=tags_filter,
                        sparse_idf=1.0,
                        idf_join="",
                        idf_filter=""
                    )

                arm_rows = await self.execute_sql(sql, params, conn=conn)
                raise TransactionRollback()

            return arm_rows

        arm_results = await asyncio.gather(*(execute_arm(search_arm) for search_arm in search_arms))
        id_lists_values: List[List[int]] = []
        weights: List[float] = []
        for search_arm, arm_rows in zip(search_arms, arm_results):
            vector_data, field_vector_id, weight, _, _ = search_arm
            id_lists_values.append([row["doc_pk_id"] for row in arm_rows])
            weights.append(weight)

            if query.raw_scores:
                fv_name = field_vector_names_by_id[field_vector_id]
                field, vector = fv_name.rsplit('_', 1)
                for rank, row in enumerate(arm_rows, start=1):
                    doc_pk_id = row["doc_pk_id"]
                    vector_score = VectorScore(
                        field=field,
                        vector=vector,
                        score=row["vdscore"],
                        rank=rank
                    )
                    if doc_pk_id not in raw_scores_map:
                        raw_scores_map[doc_pk_id] = []
                    raw_scores_map[doc_pk_id].append(vector_score)

        fused_results = self.rrf_fuse(
            id_lists=id_lists_values,
            weights=weights,
            limit=query.limit,
            score_threshold=query.score_threshold,
            k=2
        )

        top_pk_ids = [pk for pk, _ in fused_results]
        if not top_pk_ids:
            return []

        # Second stage: fetch documents for top pk_ids only
        placeholders = ", ".join(["%s"] * len(top_pk_ids))
        docs_select_sql = self.format_sql(
            "select_docs_in_with_tags",
            table=self.quote_identifier(docs_table),
            tags_table=self.quote_identifier(tags_table),
            pk_col=self.quote_identifier("pk_id"),
            id_col=self.quote_identifier("id"),
            name_col=self.quote_identifier("name"),
            description_col=self.quote_identifier("description"),
            timestamp_col=self.quote_identifier("timestamp"),
            metadata_col=self.quote_identifier("metadata"),
            tag_col=self.quote_identifier("tag"),
            tag_doc_pk_col=self.quote_identifier("doc_pk_id"),
            placeholders=placeholders
        )

        doc_rows = await self.execute_sql(docs_select_sql, tuple(top_pk_ids))
        by_pk: Dict[int, Dict[str, Any]] = {row["pk_id"]: row for row in doc_rows}

        results: List[SearchResult] = []
        for pk_id, fused_score in fused_results:
            row = by_pk.get(pk_id)
            if not row:
                continue
            meta_raw = row.get("metadata")
            if meta_raw:
                metadata = json.loads(meta_raw)
            else:
                metadata = None
            tags = self._parse_tags(row.get("tags"))
            ts = row["timestamp"]
            if ts is not None and getattr(ts, 'tzinfo', None) is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Use from_dict to handle proper type conversion
            search_result_data = {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "timestamp": ts,
                "metadata": metadata,
                "tags": tags,
                "score": fused_score,
                "vector_scores": raw_scores_map.get(pk_id, [])
            }
            results.append(SearchResult.from_dict(search_result_data, skip_validation=True))

        return results


    def _parse_tags(self, tags_raw) -> List[str]:
        """Helper method to parse tags from raw database output."""
        if tags_raw and isinstance(tags_raw, str):
            return [tag.strip() for tag in tags_raw.split("|") if tag.strip()]
        else:
            return []

    async def get_collection_info_internal(self, collection_name: str) -> CollectionConfigInternal:        
        """Get internal information about a collection."""
        # Get collection metadata from system meta table
        meta_result = await self.execute_sql(
            self.format_sql("select", 
                table=self.meta_collection,
                columns=self.quote_identifier("value"),
                where=f" WHERE {self.quote_identifier('collection_name')}=%s AND {self.quote_identifier('key')}='config'"
            ),
            (collection_name,)
        )
        
        if not meta_result:
            self.logger.debug(f"Configuration not found for collection {collection_name}")
            raise AmgixNotFound("Configuration not found")
        
        config_data = meta_result[0]["value"]
        
        # Parse JSON config
        config_dict = json.loads(config_data) if isinstance(config_data, str) else config_data
        
        # Create CollectionConfigInternal
        return CollectionConfigInternal.model_validate(config_dict)
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Union[int, Dict[str, float]]]:
        meta_result = await self.execute_sql(
            self.format_sql("select",
                table=self.meta_collection,
                columns=self.quote_identifier("value"),
                where=f" WHERE {self.quote_identifier('collection_name')}=%s AND {self.quote_identifier('key')}='stats'"
            ),
            (collection_name,)
        )
        
        if not meta_result:
            return {"doc_count": 0, "avgdls": {}}
        
        stats_data = json.loads(meta_result[0]["value"]) if isinstance(meta_result[0]["value"], str) else meta_result[0]["value"]
        return stats_data if isinstance(stats_data, dict) else {"doc_count": 0, "avgdls": {}}
    
    async def set_collection_stats(self, collection_name: str, stats: Dict[str, Union[int, Dict[str, float]]]) -> None:
        stats_json = json.dumps(stats)
        
        upsert_sql = self.format_sql("upsert",
            table=self.meta_collection,
            columns=self.quote_column_list(["collection_name", "key", "value"]),
            values="(%s, %s, %s)",
            update_clause=f"{self.quote_identifier('value')}=%s",
            conflict_columns=self.quote_column_list(["collection_name", "key"])
        )
        
        await self.execute_sql_no_result(upsert_sql, (collection_name, "stats", stats_json, stats_json))
    
    async def get_collection_info(self, collection_name: str) -> CollectionConfig:
        """Get information about a collection."""
        internal_config = await self.get_collection_info_internal(collection_name)
        
        # Convert to user-facing CollectionConfig (remove internal fields)
        return internal_to_user_config(internal_config)
    
    async def is_connected(self) -> bool:
        """
        Check if the SQL database connection is active and healthy.
        
        Returns:
            bool: True if connected and healthy, False otherwise
        """
        try:
            # Simple health check by executing a basic query
            await self.execute_sql("SELECT 1")
            return True
        except Exception as e:
            self.logger.debug(f"connection check failed: {str(e)}")
            return False
    
    # ==========================================
    # Queue Methods
    # ==========================================
    
    async def add_to_queue(self, collection_name: str, collection_id: str, documents: List[Document]) -> List[str]:
        """
        Add documents to the processing queue.
        
        Args:
            collection_name: Name of the collection these documents belong to
            collection_id: Internal collection identifier
            documents: List of documents to add to the queue
            
        Returns:
            List[str]: The queue_ids for the queue entries
        """
        if not documents:
            return []
            
        current_time = datetime.now(timezone.utc)
        rows = []
        queue_ids = []
        
        for document in documents:
            queue_id = str(uuid.uuid4())
            queue_ids.append(queue_id)
            document_json = document.model_dump_json()
            rows.append((queue_id, collection_name, collection_id, document.id, QueuedDocumentStatus.QUEUED, document_json, None, current_time, current_time, 0))
        
        # Insert in batches
        batch_size = self.DEFAULT_BATCH_SIZE
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            sql = self.generate_batch_insert_sql(
                self.queue_collection,
                ['queue_id', 'collection_name', 'collection_id', 'doc_id', 'status', 'document', 'info', 'created_at', 'timestamp', 'try_count'],
                len(batch)
            )
            params = self.flatten_batch_params(batch)
            await self.execute_sql_no_result(sql, params)
        
        return queue_ids
    
    async def get_from_queue(self, queue_ids: List[str]) -> List['QueueDocument']:
        """
        Retrieve documents from the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
            
        Returns:
            List[QueueDocument]: List of queue documents with status and metadata
        """
        # Retrieve from queue table using IN clause
        placeholders = ', '.join(['%s'] * len(queue_ids))
        result = await self.execute_sql(
            self.format_sql("select", 
                table=self.queue_collection,
                columns="*",
                where=f" WHERE {self.quote_identifier('queue_id')} IN ({placeholders})"
            ),
            tuple(queue_ids)
        )
        
        # Check if we got the expected number of results
        if len(result) != len(queue_ids):
            found_ids = {row["queue_id"] for row in result}
            missing_ids = set(queue_ids) - found_ids
            raise AmgixNotFound(f"Queue documents not found for queue_ids: {', '.join(missing_ids)}")
        
        queue_docs = []
        for row in result:
            # Parse document JSON back to Document object
            document_data = json.loads(row["document"])
            document = Document(**document_data)

            # Normalize naive datetimes to UTC-aware to avoid arithmetic errors downstream
            created_at_with_tz = row["created_at"]
            if getattr(created_at_with_tz, "tzinfo", None) is None:
                created_at_with_tz = created_at_with_tz.replace(tzinfo=timezone.utc)

            timestamp_with_tz = row["timestamp"]
            if getattr(timestamp_with_tz, "tzinfo", None) is None:
                timestamp_with_tz = timestamp_with_tz.replace(tzinfo=timezone.utc)

            # Create QueueDocument
            queue_doc = QueueDocument(
                queue_id=row["queue_id"],
                collection_name=row["collection_name"],
                collection_id=row["collection_id"],
                doc_id=row["doc_id"],
                status=row["status"],
                document=document,
                info=row["info"],
                created_at=created_at_with_tz,
                timestamp=timestamp_with_tz,
                try_count=row["try_count"]
            )
            queue_docs.append(queue_doc)
        
        return queue_docs
    
    async def delete_from_queue(self, queue_ids: List[str]) -> None:
        """
        Remove documents from the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
        """
        if not queue_ids:
            return
            
        placeholders = ','.join(['%s'] * len(queue_ids))
        await self.execute_sql_no_result(
            self.format_sql("delete", 
                table=self.queue_collection,
                where_clause=f"{self.quote_identifier('queue_id')} IN ({placeholders})"
            ),
            tuple(queue_ids)
        )
    
    async def delete_from_queue_by_collection(self, collection_name: str) -> None:
        """
        Remove all documents from the processing queue for a specific collection.
        
        Args:
            collection_name: Name of the collection to clear from queue
        """
        await self.execute_sql_no_result(
            self.format_sql("delete", 
                table=self.queue_collection,
                where_clause=f"{self.quote_identifier('collection_name')}=%s"
            ),
            (collection_name,)
        )
        

    
    async def update_queue_status(self, queue_ids: List[str], status: str, try_count: int, info: str) -> None:
        """
        Update the status of documents in the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
            status: New status to set for the documents
            try_count: New try count to set for the documents
            info: Additional information about the status (e.g., error details)
        """
        
        # Update status, timestamp, try_count, and info in the queue table using IN clause
        placeholders = ', '.join(['%s'] * len(queue_ids))
        await self.execute_sql_no_result(
            self.format_sql("update", 
                table=self.queue_collection,
                set_clause=f"{self.quote_identifier('status')}=%s, {self.quote_identifier('timestamp')}=%s, {self.quote_identifier('try_count')}=%s, {self.quote_identifier('info')}=%s",
                where_clause=f"{self.quote_identifier('queue_id')} IN ({placeholders})"
            ),
            (status, datetime.now(timezone.utc), try_count, info) + tuple(queue_ids)
        )
        
    async def get_queue_entries(self, collection_name: str, doc_id: Optional[str] = None) -> List[QueueDocument]:
        """
        Get all queue entries for a specific document or all queue entries for a collection.
        
        Args:
            collection_name: Name of the collection this document belongs to
            doc_id: Unique identifier for the document, or None to get all queue entries for the collection
            
        Returns:
            List[QueueDocument]: List of queue entries for this document or collection
        """
        # Build where clause based on whether doc_id is provided
        if doc_id is None:
            where_clause = f" WHERE {self.quote_identifier('collection_name')}=%s"
            params = (collection_name,)
        else:
            where_clause = f" WHERE {self.quote_identifier('collection_name')}=%s AND {self.quote_identifier('doc_id')}=%s"
            params = (collection_name, doc_id)
        
        # Query queue table - include doc_id in columns so we can use it in results
        # Order by doc_id and timestamp for consistent results
        result = await self.execute_sql(
            self.format_sql("select", 
                table=self.queue_collection,
                columns=self.quote_column_list(["queue_id", "collection_id", "doc_id", "status", "info", "timestamp", "created_at", "try_count"]),
                where=where_clause,
                order_by=f" ORDER BY {self.quote_identifier('timestamp')}"
            ),
            params
        )
        
        # Convert results to QueueDocument objects with only the fields we actually use
        queue_entries = []
        for row in result:
            # MariaDB returns naive datetime objects, so we need to add timezone info
            timestamp_with_tz = row["timestamp"]
            if timestamp_with_tz.tzinfo is None:
                timestamp_with_tz = timestamp_with_tz.replace(tzinfo=timezone.utc)

            # MariaDB returns naive datetime objects, so we need to add timezone info
            created_at_with_tz = row["created_at"]
            if created_at_with_tz.tzinfo is None:
                created_at_with_tz = created_at_with_tz.replace(tzinfo=timezone.utc)

            queue_doc = QueueDocument(
                queue_id=row["queue_id"],
                collection_name=collection_name,  # We know this from the method parameter
                collection_id=row["collection_id"],
                doc_id=row["doc_id"],            # Use the actual doc_id from the database
                status=row["status"],
                info=row["info"],
                document=None,                   # Not used upstream, set to None
                created_at=created_at_with_tz,   # Use actual timestamp instead of current time
                timestamp=timestamp_with_tz,
                try_count=row["try_count"]
            )
            queue_entries.append(queue_doc)
        
        return queue_entries
    
    async def get_queue_info(self, collection_name: str) -> QueueInfo:
        """
        Get queue statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            QueueInfo: Counts of documents in each queue state
        """
        # Query for counts grouped by status
        result = await self.execute_sql(
            self.format_sql("select",
                table=self.queue_collection,
                columns=f"{self.quote_identifier('status')}, COUNT(*) as count",
                where=f" WHERE {self.quote_identifier('collection_name')}=%s",
                group_by=f" GROUP BY {self.quote_identifier('status')}",
                order_by="",
                joins=""
            ),
            (collection_name,)
        )
        
        # Initialize counts for all queue statuses (excluding INDEXED)
        queue_statuses = [s for s in QueuedDocumentStatus.all() if s != QueuedDocumentStatus.INDEXED]
        counts = {status: 0 for status in queue_statuses}
        
        # Update counts from query results
        for row in result:
            status = row["status"]
            if status in counts:
                counts[status] = row["count"]
        
        total = sum(counts.values())
        
        return QueueInfo(
            queued=counts[QueuedDocumentStatus.QUEUED],
            requeued=counts[QueuedDocumentStatus.REQUEUED],
            failed=counts[QueuedDocumentStatus.FAILED],
            total=total
        )
    
    async def get_queue_statuses(self, collection_name: str, doc_id: str) -> DocumentStatusResponse:
        """
        Get comprehensive status of a document including collection and queue states.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document identifier
            
        Returns:
            DocumentStatusResponse: Complete status information with empty list if no statuses found
        """
        statuses = []
        
        # 1. Check if document is indexed in collection
        docs = (await self.get_documents(collection_name, [doc_id], suppress_not_found=True))
        if docs and docs[0] is not None:
            statuses.append(DocumentStatus(
                status=QueuedDocumentStatus.INDEXED,
                timestamp=docs[0].timestamp
            ))
        
        # 2. Get all queue entries for this doc_id
        queue_entries = await self.get_queue_entries(collection_name, doc_id)
        for entry in queue_entries:
            statuses.append(DocumentStatus(
                status=entry.status,
                info=entry.info,
                timestamp=entry.timestamp,
                queue_id=entry.queue_id,
                try_count=entry.try_count
            ))
        
        # 3. Sort by timestamp (newest first)
        statuses.sort(key=lambda x: x.timestamp, reverse=True)
        
        return DocumentStatusResponse(
            statuses=statuses
        )
        

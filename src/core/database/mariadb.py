from typing import Dict, List, Any, Optional
import asyncio
import aiomysql
from contextlib import asynccontextmanager
from urllib.parse import urlparse
import pymysql.constants

from .sql_base import SQLBase, TransactionRollback
from ..models.document import Document, DocumentWithVectors, SearchResult
from ..models.vector import CollectionConfigInternal, SearchQueryWithVectors, VectorData
from ..common import (
    SEARCH_PREFETCH_MULTIPLIER, DatabaseFeatures, DatabaseInfo, APP_PREFIX,
    DEFAULT_DB_POOL_SIZE, MIN_DB_POOL_SIZE
)


class MariaDatabase(SQLBase):
    """
    MariaDB implementation of the SQLBase interface.
    
    This class overrides SQL templates to provide MariaDB-specific syntax.
    All business logic is inherited from the base class.
    """
    
    def __init__(self, connection_string: str, logger, **kwargs):
        """Initialize MariaDB connection parameters."""
        super().__init__(connection_string, logger=logger, **kwargs)
        
        # Override SQL templates with MariaDB-specific syntax (merge with base templates)
        mariadb_templates = {
            # Table Options and Feature Flags
            "table_options": " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci",
            "if_not_exists": " IF NOT EXISTS ",
            
            # Identifier Quoting (MariaDB uses backticks)
            "quote_char": "`",
            "quote_identifier": "`{identifier}`",
            
            # Generic Table Creation Template
            "create_table": """
                CREATE TABLE {if_not_exists}`{table_name}` (
                    {columns}{indexes}
                ){table_options}
            """,
            
            # Column Definition Templates (with backticks for MariaDB)
            "column_pk": '`{name}` BIGINT AUTO_INCREMENT PRIMARY KEY',
            "column_varchar": '`{name}` VARCHAR({size}){null_constraint}',
            "column_text": '`{name}` TEXT{null_constraint}',
            "column_longtext": '`{name}` LONGTEXT{null_constraint}',
            "column_int": '`{name}` INT{null_constraint}',
            "column_bigint": '`{name}` BIGINT{null_constraint}',
            "column_float": '`{name}` FLOAT{null_constraint}',
            "column_decimal": '`{name}` DECIMAL(38,10){null_constraint}',
            "column_timestamp": '`{name}` TIMESTAMP(6){default}',
            "current_timestamp": " DEFAULT CURRENT_TIMESTAMP(6)",

            "column_vector": '`{name}` VECTOR({dimensions}) NOT NULL',
            
            # Index Definition Templates (with backticks for MariaDB)
            "index_primary": 'CONSTRAINT `pk_{name}` PRIMARY KEY (`{column}`)',
            "index_primary_multi": 'CONSTRAINT `pk_{name}` PRIMARY KEY ({columns})',
            "index_unique": 'UNIQUE KEY `ix_{name}` ({columns})',
            "index_simple": 'INDEX `ix_{name}` ({columns})',
            "index_foreign_key": 'CONSTRAINT `fk_{name}` FOREIGN KEY ({columns}) REFERENCES `{ref_table}`({ref_columns}){on_delete}',
            "index_vector": 'VECTOR INDEX `ix_{name}` (`{column}`) M=12 DISTANCE=cosine',
            "create_index_vector_cosine": 'CREATE VECTOR INDEX `ix_{name}` ON `{table}` (`{column}`) M=12 DISTANCE=cosine',
            "create_index_vector_dot": 'CREATE VECTOR INDEX `ix_{name}` ON `{table}` (`{column}`) M=12 DISTANCE=dot',
            "create_index_vector_euclid": 'CREATE VECTOR INDEX `ix_{name}` ON `{table}` (`{column}`) M=12 DISTANCE=euclidean',
            # Vector search session tuning
            "ef_search": 'SET SESSION mhnsw_ef_search = {ef_search}',
            
            # Query Templates (with backticks for MariaDB)
            "insert": "INSERT INTO `{table}` ({columns}) VALUES ({placeholders});{get_last_id}",
            "batch_insert": "INSERT INTO `{table}` ({columns}) VALUES {placeholders}",
            "update": "UPDATE `{table}` SET {set_clause} WHERE {where_clause}",
            "delete": "DELETE FROM `{table}` WHERE {where_clause}",
            "select": "SELECT {columns} FROM `{table}`{joins}{where}{group_by}{order_by}{limit}",
            "count": "SELECT COUNT(*) as count FROM `{table}`{where}",
            "drop_table": "DROP TABLE IF EXISTS {tables}",
            "alter_table_add_index": "ALTER TABLE `{table}` ADD {index_definition}",
            "create_index_simple": 'CREATE INDEX `ix_{name}` ON `{table}` ({columns})',
            "create_unique_index": 'CREATE UNIQUE INDEX `ix_{name}` ON `{table}` ({columns})',
            # Upsert/Join/Delete templates with backticked table identifiers
            "upsert": "INSERT INTO `{table}` ({columns}) VALUES {values} ON DUPLICATE KEY UPDATE {update_clause}",
            # Update clause for IDF upsert (MariaDB/MySQL)
            "upsert_update_doc_count": "doc_count = doc_count + 1",
            "update_join": "UPDATE `{table}` {alias} JOIN ({subquery}) {subquery_alias} ON {join_conditions} SET {set_clause}",
            "delete_join": "DELETE {alias} FROM `{table}` {alias} JOIN ({subquery}) {subquery_alias} ON {join_conditions} WHERE {where_clause}",
            
            # IDF Filter Template (MariaDB-specific with backticks)
            "idf_filter": "AND idf.`doc_count` <= %(idf_threshold)s",
            
            # Transaction Templates
            "last_insert_id": "SELECT LAST_INSERT_ID() as pk_id",
            # Temp tables and maintenance with backticked table identifiers
            "create_temp_table": "CREATE TEMPORARY TABLE IF NOT EXISTS `{table_name}` ({columns}){table_options}",
            "truncate": "TRUNCATE `{table}`",
            
            # Vector Search Templates (with backticks for MariaDB)
            "dense_distance_cosine": 'VEC_DISTANCE_COSINE(d.`{field_name}`, VEC_FromText(%({param_name})s))',
            "dense_distance_dot": 'VEC_DISTANCE_DOT(d.`{field_name}`, VEC_FromText(%({param_name})s))',
            "dense_distance_euclid": 'VEC_DISTANCE_EUCLIDEAN(d.`{field_name}`, VEC_FromText(%({param_name})s))',
            "dense_vector_insert": 'Vec_FromText({vector_values})',
            
            "hybrid_search": """
                SELECT vds.fv_name, d.`id`, d.`name`, d.`description`, d.`timestamp`, d.`metadata`, 
                    (SELECT GROUP_CONCAT(t.`tag` SEPARATOR '|') FROM `{tags_table}` t WHERE t.`doc_pk_id` = d.`pk_id`) tags,
                    vds.vdscore as score
                FROM `{documents_table}` d
                INNER JOIN (
                    SELECT fv_name, `doc_pk_id`, vdscore 
                    FROM (
                        {all_unions}
                    ) combined_vectors
                ) vds ON vds.`doc_pk_id` = d.`pk_id`
                ORDER BY vds.fv_name, score DESC
            """,
            
            "hybrid_scores_only": """
                SELECT fv_name, `doc_pk_id`, vdscore 
                FROM (
                    {all_unions}
                ) combined_vectors
            """,
            
            # Collection Management Templates (with backticks for MariaDB)
            "list_collections_query": """
                SELECT DISTINCT 
                    SUBSTRING(`table_name`, 1, LENGTH(`table_name`) - {docs_suffix_length}) as collection_name
                FROM information_schema.tables 
                WHERE table_schema = '{database}' 
                AND `table_name` LIKE '{prefix}' 
                AND `table_name` LIKE '%_{docs_suffix}'
            """,
            
            # Database Probing Templates (with backticks for MariaDB)
            "version_query": "SELECT VERSION()",
            "vector_test_create": f"CREATE TEMPORARY TABLE `{APP_PREFIX}_test_vector_idx` (v VECTOR(1) NOT NULL)",
            "vector_test_drop": f"DROP TEMPORARY TABLE `{APP_PREFIX}_test_vector_idx`",
            
            # Document Type Filtering Templates (with backticks for MariaDB)
            "tags_filter": "WHERE d.pk_id IN (SELECT doc_pk_id FROM `{tags_table}` t WHERE t.doc_pk_id = d.pk_id AND t.tag IN %(document_tags)s)",
            "tags_filter_and": "WHERE d.pk_id IN (SELECT doc_pk_id FROM `{tags_table}` t WHERE t.doc_pk_id = d.pk_id AND t.tag IN %(document_tags)s GROUP BY doc_pk_id HAVING COUNT(DISTINCT t.tag) = %(document_tags_count)s)",
            "sparse_documents_join": "INNER JOIN `{documents_table}` d ON d.`pk_id` = vd.`doc_pk_id`",
            # "sparse_idf": "(LN((SELECT COUNT(*) FROM `{documents_table}`) + 1 / (SELECT COUNT(DISTINCT doc_pk_id) + 0.5 FROM `{vector_data_table}` vd_idf WHERE vd_idf.`token_id` = vd.`token_id` AND vd_idf.`field_vector_name` = vd.`field_vector_name`)))",
            "sparse_idf": "LN((%(total_docs)s + 1) / (idf.doc_count + 0.5))",
            # "sparse_idf": """LOG(
            #     ((SELECT COUNT(*) FROM `{documents_table}`) -
            #     (SELECT COUNT(DISTINCT doc_pk_id) FROM `{vector_data_table}` vd_idf WHERE vd_idf.`token_id` = vd.`token_id`)) /
            #     (SELECT COUNT(DISTINCT doc_pk_id) FROM `{vector_data_table}` vd_idf WHERE vd_idf.`token_id` = vd.`token_id`))""",
            
            # Sparse Vector Search Templates (with backticks for MariaDB)
            "sparse_union_single": """
                SELECT '{field_vector_name}' fv_name, `doc_pk_id`, vdscore FROM (
                    SELECT '{field_vector_name}' fv_name, vd.`doc_pk_id`, SUM(vd.`weight` * qv.`weight` * {sparse_idf}) vdscore
                      FROM `{vector_data_table}` vd
                      {sparse_documents_join}
                      INNER JOIN `{query_vectors_table}` qv ON qv.`field_vector_name` = vd.`field_vector_name` 
                        AND qv.`token_id` = vd.`token_id` 
                        AND qv.`field_vector_name` = '{field_vector_name}'
                      {idf_join}
                      {tags_filter}
                      {idf_filter}
                      GROUP BY fv_name, vd.`doc_pk_id`
                      ORDER BY vdscore DESC 
                      LIMIT %(prefetch_limit)s
                ) sparse_subquery""",
            
            # Dense Vector Search Templates (with backticks for MariaDB)
            "dense_union_single": """
                SELECT '{field_vector_name}' fv_name, `pk_id` `doc_pk_id`, vdscore FROM (
                    SELECT '{field_vector_name}' fv_name, d.`pk_id`, (1 - {distance_expr}) vdscore
                      FROM `{documents_table}` d 
                      {tags_filter}
                      ORDER BY {distance_expr} 
                      LIMIT %(prefetch_limit)s
                ) dense_subquery""",
        }
        
        # Merge MariaDB-specific templates with base templates
        self.SQL_TEMPLATES.update(mariadb_templates)
        
        # # Parse connection string using urllib.parse
        # if not connection_string.startswith('mariadb://'):
        #     raise ValueError("Connection string must start with 'mariadb://'")
        
        parsed = urlparse(connection_string)
        
        # Extract components
        self.host = parsed.hostname
        self.port = parsed.port or 3306
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
        self.charset = 'utf8mb4'
    
    async def _get_connection(self):
        """Get a connection from the pool."""
        if self.pool is None:
            # Lazy-connect to avoid requiring explicit probe/connect in callers
            await self.connect()
        return await self.pool.acquire()

    async def create_collection(self, collection_name: str, config: CollectionConfigInternal) -> bool:
        """Create collection via base implementation; cleanup if DDL partially applied.

        MariaDB doesn't rollback DDL in transactions. If the base method fails,
        we drop any possibly created tables to leave the schema consistent.
        """
        try:
            return await super().create_collection(collection_name, config)
        except Exception:
            # Cleanup best-effort: drop any partially created tables, then re-raise
            try:
                await self.delete_collection(collection_name)
            except Exception as cleanup_error:
                # Log cleanup errors but don't let them mask the original error
                self.logger.warning(f"Failed to cleanup collection {collection_name} after creation failure: {cleanup_error}")
            raise

    
    async def connect(self):
        """Establish connection pool to the database."""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                charset=self.charset,
                autocommit=True,
                minsize=MIN_DB_POOL_SIZE,
                maxsize=DEFAULT_DB_POOL_SIZE,
                client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS,
                # Set transaction isolation level to READ COMMITTED to reduce deadlocks
                init_command="SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    async def execute_sql(self, sql: str, params: Optional[tuple] = None, skip_first: bool = False, conn=None) -> List[Dict[str, Any]]:
        """Execute SQL and return results as list of dictionaries."""
        # Use provided connection or get one from the pool
        connection_provided = conn is not None
        if not connection_provided:
            conn = await self._get_connection()
        
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, params)
                if skip_first:
                    await cursor.nextset()
                results = await cursor.fetchall()
                return [dict(row) for row in results]
        finally:
            # Only release if we got the connection ourselves
            if not connection_provided:
                self.pool.release(conn)
    
    async def execute_sql_no_result(self, sql: str, params: Optional[tuple] = None, conn=None) -> None:
        """Execute SQL that doesn't return results (INSERT, UPDATE, DELETE)."""
        # Use provided connection or get one from the pool
        connection_provided = conn is not None
        if not connection_provided:
            conn = await self._get_connection()
        
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, params)
        finally:
            # Only release if we got the connection ourselves
            if not connection_provided:
                self.pool.release(conn)
    

    
    # ==========================================
    # Transaction Management (MariaDB)
    # ==========================================
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for transactions that ensures proper commit/rollback."""
        conn = await self._get_connection()
        try:
            # Disable autocommit for this connection during the transaction
            await conn.begin()
            
            # Yield the connection to be used in the context block
            yield conn
            
            # If we get here without exceptions, commit the transaction
            await conn.commit()
        except Exception as e:
            # On any error, rollback the transaction
            await conn.rollback()
            if isinstance(e, TransactionRollback):
                # Do not re-raise for controlled rollback-on-success
                return
            raise
        finally:
            # Always return the connection to the pool
            self.pool.release(conn)
    
    


"""Database Connector component for the Agent Builder.

Reads connection details from the Connectors Catalogue (configured via the
Connectors page) and provides connection parameters + schema metadata to
downstream components such as Talk-to-Data (NL→SQL).

Follows the same pattern as RegistryModelComponent — a dropdown populated from
the catalogue table, no manual connection fields.
"""

import asyncio
import concurrent.futures
import os
import threading

from agentcore.custom.custom_node.node import Node
from agentcore.inputs.inputs import BoolInput, DropdownInput, MultilineInput
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.logging import logger


# ---------------------------------------------------------------------------
# Shared sync engine (same approach as registry_model.py)
# ---------------------------------------------------------------------------
_sync_engine = None
_sync_engine_lock = threading.Lock()


def _get_sync_engine():
    """Return a dedicated synchronous SQLAlchemy engine (created once)."""
    global _sync_engine
    if _sync_engine is not None:
        return _sync_engine

    with _sync_engine_lock:
        if _sync_engine is not None:
            return _sync_engine

        from sqlalchemy import create_engine
        from agentcore.services.deps import get_db_service

        db_service = get_db_service()
        db_url = db_service.database_url
        if "+asyncpg" in db_url:
            db_url = db_url.replace("+asyncpg", "")

        _sync_engine = create_engine(db_url, pool_pre_ping=True, pool_size=3)
        logger.info(f"Created sync engine for DatabaseConnector: {db_url.split('@')[-1]}")
        return _sync_engine


def _run_async(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Catalogue data access
# ---------------------------------------------------------------------------

def _fetch_connectors() -> list[str]:
    """Fetch all connectors from the catalogue.

    Returns list of strings: 'name | provider | host:port/database | uuid'
    """
    try:
        from agentcore.services.deps import get_db_service

        db_service = get_db_service()

        async def _query():
            from sqlalchemy import select
            from agentcore.services.database.models.connector_catalogue.model import ConnectorCatalogue

            async with db_service.with_session() as session:
                stmt = (
                    select(ConnectorCatalogue)
                    .where(ConnectorCatalogue.status == "connected")
                    .order_by(ConnectorCatalogue.name)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    f"{r.name} | {r.provider} | {r.host}:{r.port}/{r.database_name} | {r.id}"
                    for r in rows
                ]

        return _run_async(_query())
    except Exception as e:
        logger.warning(f"Could not fetch connectors from catalogue: {e}")
        return []


def _get_connector_config(connector_id: str) -> dict | None:
    """Fetch connector config by ID using sync engine, with decrypted password."""
    from uuid import UUID
    from sqlalchemy.orm import Session
    from agentcore.services.database.models.connector_catalogue.model import ConnectorCatalogue

    try:
        engine = _get_sync_engine()
        with Session(engine) as session:
            row = session.get(ConnectorCatalogue, UUID(connector_id))
            if row is None:
                logger.warning(f"Connector {connector_id} not found in catalogue")
                return None

            # Use the SAME decrypt function as the API — guarantees same Fernet key
            password = ""
            if row.password_encrypted:
                try:
                    from agentcore.api.connector_catalogue import _decrypt_password
                    password = _decrypt_password(row.password_encrypted)
                except Exception as e:
                    logger.error(f"Failed to decrypt connector password: {e}")

            return {
                "provider": row.provider,
                "host": row.host,
                "port": row.port,
                "database_name": row.database_name,
                "schema_name": row.schema_name or "public",
                "username": row.username,
                "password": password,
                "ssl_enabled": row.ssl_enabled,
                "tables_metadata": row.tables_metadata,
            }
    except Exception as e:
        logger.error(f"Failed to fetch connector config for {connector_id}: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class DatabaseConnectorComponent(Node):
    """Select a database from the Connectors Catalogue and expose its schema.

    Connection details (host, port, credentials, etc.) are configured once on
    the Connectors page.  In the Agent Builder you simply pick which connector
    to use from the dropdown.
    """

    display_name = "Database Connector"
    description = "Select a database from the Connectors Catalogue and expose its schema for downstream components."
    icon = "database"
    name = "DatabaseConnector"

    inputs = [
        DropdownInput(
            name="connector",
            display_name="Connector",
            info="Select a connector from the Connectors Catalogue. Configure connectors on the Connectors page.",
            options=[],
            value="",
            refresh_button=True,
            real_time_refresh=True,
            combobox=True,
        ),
        MultilineInput(
            name="tables_filter",
            display_name="Tables Filter (comma-separated)",
            value="",
            info="Comma-separated list of table names to include. Leave empty to include all tables.",
            advanced=True,
        ),
        BoolInput(
            name="discover_fks",
            display_name="Auto-discover Foreign Keys",
            value=True,
            info="Automatically discover foreign key relationships and include them in the schema DDL.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Connection Config",
            name="connection_config",
            method="provide_connection",
            types=["Data"],
        ),
        Output(
            display_name="Schema Info",
            name="schema_info",
            method="provide_schema",
            types=["Message"],
        ),
    ]

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        """Refresh the connector dropdown from the Connectors Catalogue."""
        if field_name == "connector":
            try:
                options = _fetch_connectors()
                build_config["connector"]["options"] = options if options else []
                # Keep current value if still valid, otherwise pick first
                current = build_config["connector"].get("value", "")
                if current not in options:
                    build_config["connector"]["value"] = options[0] if options else ""
            except Exception as e:
                logger.warning(f"Error fetching connectors: {e}")
                build_config["connector"]["options"] = []
        return build_config

    def _get_selected_config(self) -> dict:
        """Parse the selected connector dropdown value and fetch config from DB."""
        selected = self.connector
        if not selected:
            msg = "No connector selected. Please select a connector from the dropdown."
            raise ValueError(msg)

        # Parse: "name | provider | host:port/database | uuid"
        parts = [p.strip() for p in selected.split("|")]
        if len(parts) < 4:
            msg = f"Invalid connector format: {selected}. Please refresh the dropdown."
            raise ValueError(msg)

        connector_id = parts[3]
        config = _get_connector_config(connector_id)
        if config is None:
            msg = f"Connector '{parts[0]}' not found or has been deleted. Please refresh."
            raise ValueError(msg)

        return config

    def _fetch_schema(self, params: dict) -> list[dict]:
        """Connect to database and fetch table/column metadata."""
        provider = params["provider"]

        if provider == "postgresql":
            import psycopg2

            conn_kwargs = {
                "host": params["host"],
                "port": params["port"],
                "dbname": params["database_name"],
                "user": params["username"],
                "password": params["password"],
                "connect_timeout": 15,
            }
            if params.get("ssl_enabled"):
                conn_kwargs["sslmode"] = "require"

            conn = psycopg2.connect(**conn_kwargs)
            cur = conn.cursor()

            schema = params.get("schema_name", "public")

            cur.execute("""
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name NOT LIKE 'pg_%%'
                ORDER BY table_name, ordinal_position
            """, (schema,))
            rows = cur.fetchall()

            tables = {}
            for tbl, col, dtype, nullable in rows:
                if tbl not in tables:
                    tables[tbl] = {"table_name": tbl, "columns": []}
                tables[tbl]["columns"].append({
                    "name": col,
                    "type": dtype,
                    "nullable": nullable == "YES",
                })

            cur.close()
            conn.close()

            # Apply table filter if provided
            filter_str = self.tables_filter.strip() if self.tables_filter else ""
            if filter_str:
                allowed = {t.strip().lower() for t in filter_str.split(",") if t.strip()}
                tables = {k: v for k, v in tables.items() if k.lower() in allowed}

            return list(tables.values())

        elif provider == "oracle":
            import oracledb

            dsn = oracledb.makedsn(
                params["host"], params["port"],
                service_name=params["database_name"],
            )
            conn = oracledb.connect(
                user=params["username"],
                password=params["password"],
                dsn=dsn,
            )
            cur = conn.cursor()

            cur.execute("""
                SELECT table_name, column_name, data_type,
                       CASE WHEN nullable = 'Y' THEN 'YES' ELSE 'NO' END AS is_nullable
                FROM user_tab_columns
                WHERE table_name NOT LIKE 'BIN$%'
                ORDER BY table_name, column_id
            """)
            rows = cur.fetchall()

            tables = {}
            for tbl, col, dtype, nullable in rows:
                if tbl not in tables:
                    tables[tbl] = {"table_name": tbl, "columns": []}
                tables[tbl]["columns"].append({
                    "name": col,
                    "type": dtype,
                    "nullable": nullable == "YES",
                })

            cur.close()
            conn.close()

            # Apply table filter if provided
            filter_str = self.tables_filter.strip() if self.tables_filter else ""
            if filter_str:
                allowed = {t.strip().upper() for t in filter_str.split(",") if t.strip()}
                tables = {k: v for k, v in tables.items() if k.upper() in allowed}

            return list(tables.values())
        else:
            logger.warning(f"Provider '{provider}' not yet supported for schema fetch")
            return []

    def _fetch_foreign_keys(self, params: dict) -> list[dict]:
        """Fetch foreign key relationships from database metadata."""
        provider = params.get("provider", "")
        if provider not in ("postgresql", "oracle"):
            return []

        if provider == "oracle":
            return self._fetch_oracle_foreign_keys(params)

        import psycopg2

        conn_kwargs = {
            "host": params["host"],
            "port": params["port"],
            "dbname": params["database_name"],
            "user": params["username"],
            "password": params["password"],
            "connect_timeout": 15,
        }
        if params.get("ssl_enabled"):
            conn_kwargs["sslmode"] = "require"

        conn = psycopg2.connect(**conn_kwargs)
        cur = conn.cursor()

        schema = params.get("schema_name", "public")

        cur.execute("""
            SELECT
                tc.table_name       AS source_table,
                kcu.column_name     AS source_column,
                ccu.table_name      AS target_table,
                ccu.column_name     AS target_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON tc.constraint_name = ccu.constraint_name
                AND tc.table_schema = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = %s
            ORDER BY tc.table_name, kcu.column_name
        """, (schema,))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Apply table filter if provided
        filter_str = self.tables_filter.strip() if self.tables_filter else ""
        allowed = None
        if filter_str:
            allowed = {t.strip().lower() for t in filter_str.split(",") if t.strip()}

        results = []
        for source_table, source_column, target_table, target_column in rows:
            if allowed and source_table.lower() not in allowed:
                continue
            results.append({
                "source_table": source_table,
                "source_column": source_column,
                "target_table": target_table,
                "target_column": target_column,
            })

        return results

    def _fetch_oracle_foreign_keys(self, params: dict) -> list[dict]:
        """Fetch foreign key relationships from Oracle data dictionary."""
        import oracledb

        dsn = oracledb.makedsn(
            params["host"], params["port"],
            service_name=params["database_name"],
        )
        conn = oracledb.connect(
            user=params["username"],
            password=params["password"],
            dsn=dsn,
        )
        cur = conn.cursor()

        cur.execute("""
            SELECT
                a.table_name       AS source_table,
                a.column_name      AS source_column,
                c_pk.table_name    AS target_table,
                b.column_name      AS target_column
            FROM user_cons_columns a
            JOIN user_constraints c ON a.constraint_name = c.constraint_name
            JOIN user_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
            JOIN user_cons_columns b ON c_pk.constraint_name = b.constraint_name
                AND a.position = b.position
            WHERE c.constraint_type = 'R'
            ORDER BY a.table_name, a.column_name
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Apply table filter if provided
        filter_str = self.tables_filter.strip() if self.tables_filter else ""
        allowed = None
        if filter_str:
            allowed = {t.strip().upper() for t in filter_str.split(",") if t.strip()}

        results = []
        for source_table, source_column, target_table, target_column in rows:
            if allowed and source_table.upper() not in allowed:
                continue
            results.append({
                "source_table": source_table,
                "source_column": source_column,
                "target_table": target_table,
                "target_column": target_column,
            })

        return results

    def _build_ddl_string(self, tables_meta: list[dict], fk_list: list[dict] | None = None) -> str:
        """Convert schema metadata into DDL-like string for LLM context."""
        # Index FKs by source table for fast lookup
        fk_by_table: dict[str, list[dict]] = {}
        if fk_list:
            for fk in fk_list:
                fk_by_table.setdefault(fk["source_table"], []).append(fk)

        ddl_parts = []
        for table in tables_meta:
            cols = []
            for col in table["columns"]:
                null_str = "" if col["nullable"] else " NOT NULL"
                cols.append(f"  {col['name']} {col['type'].upper()}{null_str}")

            # Append FK relationships as comments
            table_fks = fk_by_table.get(table["table_name"], [])
            fk_comments = []
            for fk in table_fks:
                fk_comments.append(
                    f"  -- FK: {fk['source_column']} REFERENCES "
                    f"{fk['target_table']}({fk['target_column']})"
                )

            ddl = f"TABLE {table['table_name']} (\n" + ",\n".join(cols)
            if fk_comments:
                ddl += "\n" + "\n".join(fk_comments)
            ddl += "\n)"
            ddl_parts.append(ddl)
        return "\n\n".join(ddl_parts)

    def provide_connection(self) -> Data:
        """Output the connection config + schema as structured Data."""
        try:
            params = self._get_selected_config()
            logger.info(f"DatabaseConnector: loaded config for {params['provider']}://{params['host']}:{params['port']}/{params['database_name']}")
        except Exception as e:
            self.status = f"Config error: {e!s}"
            logger.error(f"DatabaseConnector config error: {e!s}", exc_info=True)
            return Data(data={"status": f"error: {e!s}", "schema_ddl": ""})

        try:
            tables_meta = self._fetch_schema(params)

            # FK discovery (non-fatal)
            fk_list = []
            if self.discover_fks:
                try:
                    fk_list = self._fetch_foreign_keys(params)
                    logger.info(f"DatabaseConnector: discovered {len(fk_list)} FK relationships")
                except Exception as e:
                    logger.warning(f"FK discovery failed (non-fatal): {e}")

            schema_ddl = self._build_ddl_string(tables_meta, fk_list)
            status = "connected"
            table_count = len(tables_meta)
            self.status = f"Connected: {table_count} tables, {len(fk_list)} FKs"
            logger.info(f"DatabaseConnector: fetched schema with {table_count} tables, DDL length={len(schema_ddl)}")
        except Exception as e:
            tables_meta = []
            fk_list = []
            schema_ddl = ""
            status = f"error: {e!s}"
            table_count = 0
            self.status = f"Connection error: {e!s}"
            logger.error(f"DatabaseConnector connection/schema error: {e!s}", exc_info=True)

        return Data(data={
            "provider": params["provider"],
            "host": params["host"],
            "port": params["port"],
            "database_name": params["database_name"],
            "schema_name": params["schema_name"],
            "username": params["username"],
            "password": params["password"],
            "ssl_enabled": params["ssl_enabled"],
            "status": status,
            "tables_metadata": tables_meta,
            "foreign_keys": fk_list,
            "schema_ddl": schema_ddl,
            "table_count": table_count,
        })

    def provide_schema(self) -> Message:
        """Output a human-readable schema description."""
        try:
            params = self._get_selected_config()
        except Exception as e:
            self.status = f"Config error: {e!s}"
            logger.error(f"DatabaseConnector config error (schema): {e!s}", exc_info=True)
            return Message(text=f"Failed to load connector config: {e!s}")

        try:
            tables_meta = self._fetch_schema(params)

            # FK discovery (non-fatal)
            fk_list = []
            if self.discover_fks:
                try:
                    fk_list = self._fetch_foreign_keys(params)
                except Exception as e:
                    logger.warning(f"FK discovery failed (non-fatal, schema output): {e}")

            ddl = self._build_ddl_string(tables_meta, fk_list)
            summary = f"**Database Schema** ({params['provider']}://{params['host']}:{params['port']}/{params['database_name']})\n\n"
            summary += f"Schema: `{params['schema_name']}` | Tables: **{len(tables_meta)}** | FKs: **{len(fk_list)}**\n\n"
            summary += f"```sql\n{ddl}\n```"
            self.status = f"Schema: {len(tables_meta)} tables, {len(fk_list)} FKs"
            return Message(text=summary)
        except Exception as e:
            self.status = f"Error: {e!s}"
            return Message(text=f"Failed to fetch schema: {e!s}")
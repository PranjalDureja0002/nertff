from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from agentcore.api.utils import CurrentActiveUser, DbSession
from agentcore.services.auth.permissions import get_permissions_for_role, normalize_role
from agentcore.services.database.models.department.model import Department
from agentcore.services.database.models.organization.model import Organization
from agentcore.services.database.models.user.model import User
from agentcore.services.database.models.user_department_membership.model import UserDepartmentMembership
from agentcore.services.database.models.user_organization_membership.model import UserOrganizationMembership
from agentcore.services.database.models.connector_catalogue.model import ConnectorCatalogue

router = APIRouter(prefix="/connector-catalogue", tags=["Connector Catalogue"])

DB_PROVIDERS = {"postgresql", "oracle", "sqlserver", "mysql"}
STORAGE_PROVIDERS = {"azure_blob", "sharepoint"}


# ---------- Encryption helpers ----------

_FERNET_KEY = None


def _derive_encryption_key() -> str:
    """Derive a deterministic Fernet key from WEBUI_SECRET_KEY.

    This ensures the same key is used across server restarts, so
    previously encrypted passwords remain decryptable.
    Same approach as Model Registry (registry_model.py).
    """
    import base64
    import hashlib
    import os

    # Allow explicit override via env var
    explicit = os.getenv("CONNECTOR_ENCRYPTION_KEY", "")
    if explicit:
        return explicit

    # Derive deterministically from the platform secret
    raw = os.getenv("WEBUI_SECRET_KEY", "default-agentcore-connector-key")
    derived = hashlib.sha256(raw.encode()).digest()
    return base64.urlsafe_b64encode(derived).decode()


def _get_fernet():
    global _FERNET_KEY
    if _FERNET_KEY is None:
        from cryptography.fernet import Fernet

        key = _derive_encryption_key()
        _FERNET_KEY = Fernet(key.encode() if isinstance(key, str) else key)
    return _FERNET_KEY


def _encrypt_password(password: str) -> str:
    return _get_fernet().encrypt(password.encode()).decode()


def _decrypt_password(encrypted: str) -> str:
    return _get_fernet().decrypt(encrypted.encode()).decode()


def _encrypt_provider_config(provider: str, config: dict) -> dict:
    """Encrypt sensitive fields in provider_config before saving."""
    encrypted = dict(config)
    if provider == "azure_blob" and "connection_string" in encrypted:
        encrypted["connection_string"] = _encrypt_password(encrypted["connection_string"])
    elif provider == "sharepoint" and "client_secret" in encrypted:
        encrypted["client_secret"] = _encrypt_password(encrypted["client_secret"])
    return encrypted


def _decrypt_provider_config(provider: str, config: dict) -> dict:
    """Decrypt sensitive fields in provider_config when reading."""
    decrypted = dict(config)
    try:
        if provider == "azure_blob" and "connection_string" in decrypted:
            decrypted["connection_string"] = _decrypt_password(decrypted["connection_string"])
        elif provider == "sharepoint" and "client_secret" in decrypted:
            decrypted["client_secret"] = _decrypt_password(decrypted["client_secret"])
    except Exception:
        pass
    return decrypted


# ---------- Payloads ----------

class ConnectorPayload(BaseModel):
    name: str
    description: str | None = None
    provider: str  # postgresql | oracle | sqlserver | mysql | azure_blob | sharepoint
    # DB-only fields (optional for non-DB providers)
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    schema_name: str = "public"
    username: str | None = None
    password: str | None = None
    ssl_enabled: bool = False
    # Non-DB provider config (Azure Blob, SharePoint)
    provider_config: dict | None = None
    is_custom: bool = False
    org_id: UUID | None = None
    dept_id: UUID | None = None
    visibility: str = "private"  # private | public
    public_scope: str | None = None  # organization | department (required when visibility=public)
    public_dept_ids: list[UUID] | None = None  # super_admin can select multiple departments
    shared_user_emails: list[str] | None = None  # optional for department_admin when private


class ConnectorUpdatePayload(BaseModel):
    name: str | None = None
    description: str | None = None
    provider: str | None = None
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    schema_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl_enabled: bool | None = None
    provider_config: dict | None = None
    is_custom: bool | None = None
    org_id: UUID | None = None
    dept_id: UUID | None = None
    visibility: str | None = None
    public_scope: str | None = None
    public_dept_ids: list[UUID] | None = None
    shared_user_emails: list[str] | None = None


class TestConnectionPayload(BaseModel):
    provider: str | None = None
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    schema_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl_enabled: bool | None = None
    provider_config: dict | None = None


# ---------- RBAC helpers (same pattern as VectorDB) ----------

def _is_root_user(current_user: CurrentActiveUser) -> bool:
    return str(getattr(current_user, "role", "")).lower() == "root"


async def _require_connector_permission(current_user: CurrentActiveUser, permission: str) -> None:
    user_permissions = await get_permissions_for_role(str(current_user.role))
    if permission not in user_permissions:
        raise HTTPException(status_code=403, detail="Missing required permissions")


def _normalize_visibility(value: str | None) -> str:
    normalized = (value or "private").strip().lower()
    if normalized not in {"private", "public"}:
        raise HTTPException(status_code=400, detail=f"Unsupported visibility '{value}'")
    return normalized


def _normalize_public_scope(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {"organization", "department"}:
        raise HTTPException(status_code=400, detail=f"Unsupported public_scope '{value}'")
    return normalized


async def _get_scope_memberships(session: DbSession, user_id: UUID) -> tuple[set[UUID], list[tuple[UUID, UUID]]]:
    org_rows = (
        await session.exec(
            select(UserOrganizationMembership.org_id).where(
                UserOrganizationMembership.user_id == user_id,
                UserOrganizationMembership.status.in_(["accepted", "active"]),
            )
        )
    ).all()
    dept_rows = (
        await session.exec(
            select(UserDepartmentMembership.org_id, UserDepartmentMembership.department_id).where(
                UserDepartmentMembership.user_id == user_id,
                UserDepartmentMembership.status == "active",
            )
        )
    ).all()
    org_ids = {r if isinstance(r, UUID) else r[0] for r in org_rows}
    return org_ids, [(row[0], row[1]) for row in dept_rows]


def _string_ids(values: list[UUID] | None) -> list[str]:
    return [str(v) for v in (values or [])]


async def _validate_scope_refs(session: DbSession, org_id: UUID | None, dept_id: UUID | None) -> None:
    if dept_id and not org_id:
        raise HTTPException(status_code=400, detail="dept_id requires org_id")
    if org_id:
        org = await session.get(Organization, org_id)
        if not org:
            raise HTTPException(status_code=400, detail="Invalid org_id")
    if dept_id:
        dept = (
            await session.exec(
                select(Department).where(Department.id == dept_id, Department.org_id == org_id)
            )
        ).first()
        if not dept:
            raise HTTPException(status_code=400, detail="Invalid dept_id for org_id")


async def _ensure_connector_name_available(
    session: DbSession,
    name: str,
    org_id: UUID | None,
    dept_id: UUID | None,
    *,
    exclude_id: UUID | None = None,
) -> None:
    stmt = select(ConnectorCatalogue.id).where(
        func.lower(ConnectorCatalogue.name) == name.strip().lower(),
    )
    stmt = stmt.where(
        ConnectorCatalogue.org_id.is_(None) if org_id is None else ConnectorCatalogue.org_id == org_id,
    )
    stmt = stmt.where(
        ConnectorCatalogue.dept_id.is_(None) if dept_id is None else ConnectorCatalogue.dept_id == dept_id,
    )
    if exclude_id:
        stmt = stmt.where(ConnectorCatalogue.id != exclude_id)
    existing = (await session.exec(stmt)).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Connector name already exists for this scope")


# ---------- Serialization ----------

def _serialize_connector(row: ConnectorCatalogue) -> dict:
    # Return provider_config with secrets masked (not decrypted) for display
    safe_config: dict | None = None
    if row.provider_config:
        safe_config = dict(row.provider_config)
        if row.provider == "azure_blob" and "connection_string" in safe_config:
            safe_config["connection_string"] = "********"
        elif row.provider == "sharepoint" and "client_secret" in safe_config:
            safe_config["client_secret"] = "********"

    return {
        "id": str(row.id),
        "name": row.name,
        "description": row.description or "",
        "provider": row.provider,
        "host": row.host,
        "port": row.port,
        "database_name": row.database_name,
        "schema_name": row.schema_name,
        "username": row.username,
        "ssl_enabled": row.ssl_enabled,
        "provider_config": safe_config,
        "status": row.status,
        "tables_metadata": row.tables_metadata,
        "last_tested_at": row.last_tested_at.isoformat() if row.last_tested_at else None,
        "isCustom": bool(row.is_custom),
        "org_id": str(row.org_id) if row.org_id else None,
        "dept_id": str(row.dept_id) if row.dept_id else None,
        "visibility": row.visibility,
        "public_scope": row.public_scope,
        "public_dept_ids": row.public_dept_ids or [],
        "shared_user_ids": row.shared_user_ids or [],
    }


async def _resolve_user_ids_by_emails(session: DbSession, emails: list[str]) -> list[str]:
    if not emails:
        return []
    normalized = [e.strip().lower() for e in emails if e and e.strip()]
    if not normalized:
        return []
    rows = (
        await session.exec(select(User.id, User.email).where(User.email.in_(normalized)))
    ).all()
    found = {str(r[1]).lower(): str(r[0]) for r in rows}
    missing = [e for e in normalized if e not in found]
    if missing:
        raise HTTPException(status_code=400, detail=f"Invalid shared_user_emails: {', '.join(missing)}")
    return [found[e] for e in normalized]


async def _validate_departments_exist_for_org(session: DbSession, org_id: UUID, dept_ids: list[UUID]) -> None:
    if not dept_ids:
        return
    rows = (
        await session.exec(
            select(Department.id).where(Department.org_id == org_id, Department.id.in_(dept_ids))
        )
    ).all()
    if len({str(r if isinstance(r, UUID) else r[0]) for r in rows}) != len({str(d) for d in dept_ids}):
        raise HTTPException(status_code=400, detail="One or more public_dept_ids are invalid for org_id")


async def _enforce_creation_scope(
    session: DbSession,
    current_user: CurrentActiveUser,
    payload: ConnectorPayload | ConnectorUpdatePayload,
) -> tuple[str, str | None, list[str], list[str]]:
    user_role = normalize_role(str(current_user.role))
    visibility = _normalize_visibility(getattr(payload, "visibility", None))
    public_scope = _normalize_public_scope(getattr(payload, "public_scope", None))
    public_dept_ids = _string_ids(getattr(payload, "public_dept_ids", None))
    shared_user_ids: list[str] = []
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    dept_ids = {dept_id for _, dept_id in dept_pairs}

    if user_role not in {"root", "super_admin", "department_admin", "developer", "business_user"}:
        raise HTTPException(status_code=403, detail="Your role is not allowed to create connectors")

    if visibility == "private":
        payload.public_scope = None
        payload.public_dept_ids = None
        if user_role == "department_admin":
            if not dept_pairs:
                raise HTTPException(status_code=403, detail="No active department scope found")
            # private for department admin is scoped to current department
            current_org_id, current_dept_id = sorted(dept_pairs, key=lambda x: (str(x[0]), str(x[1])))[0]
            payload.org_id = current_org_id
            payload.dept_id = current_dept_id
            shared_user_ids = await _resolve_user_ids_by_emails(session, getattr(payload, "shared_user_emails", None) or [])
            if shared_user_ids:
                allowed_ids = set(
                    str(v if isinstance(v, UUID) else v[0])
                    for v in (
                        await session.exec(
                            select(UserDepartmentMembership.user_id).where(
                                UserDepartmentMembership.department_id == current_dept_id,
                                UserDepartmentMembership.status == "active",
                            )
                        )
                    ).all()
                )
                if not set(shared_user_ids).issubset(allowed_ids):
                    raise HTTPException(status_code=403, detail="shared_user_emails must belong to your current department")
        else:
            if user_role in {"developer", "business_user"} and dept_pairs:
                current_org_id, current_dept_id = sorted(dept_pairs, key=lambda x: (str(x[0]), str(x[1])))[0]
                payload.org_id = current_org_id
                payload.dept_id = current_dept_id
            else:
                payload.org_id = None
                payload.dept_id = None
    else:
        # public
        if public_scope is None:
            raise HTTPException(status_code=400, detail="public_scope is required when visibility is public")
        if public_scope == "organization":
            if not payload.org_id:
                raise HTTPException(status_code=400, detail="org_id is required for public organization visibility")
            if user_role != "root" and payload.org_id not in org_ids:
                raise HTTPException(status_code=403, detail="org_id must belong to your organization scope")
            payload.dept_id = None
            payload.public_dept_ids = None
            public_dept_ids = []
        else:
            if user_role in {"super_admin", "root"}:
                if not payload.org_id:
                    raise HTTPException(status_code=400, detail="org_id is required for department visibility")
                if user_role != "root" and payload.org_id not in org_ids:
                    raise HTTPException(status_code=403, detail="org_id must belong to your organization scope")
                if not public_dept_ids and payload.dept_id:
                    public_dept_ids = [str(payload.dept_id)]
                if not public_dept_ids:
                    raise HTTPException(status_code=400, detail="Select at least one department")
                await _validate_departments_exist_for_org(session, payload.org_id, [UUID(v) for v in public_dept_ids])
                payload.dept_id = UUID(public_dept_ids[0]) if len(public_dept_ids) == 1 else None
            else:
                if not dept_pairs:
                    raise HTTPException(status_code=403, detail="No active department scope found")
                # non-super-admin can only publish to their own current department
                current_org_id, current_dept_id = sorted(dept_pairs, key=lambda x: (str(x[0]), str(x[1])))[0]
                payload.org_id = current_org_id
                payload.dept_id = current_dept_id
                public_dept_ids = [str(current_dept_id)]
        shared_user_ids = []

    await _validate_scope_refs(session, payload.org_id, payload.dept_id)
    return visibility, public_scope, public_dept_ids, shared_user_ids


def _can_access_connector(
    row: ConnectorCatalogue,
    current_user: CurrentActiveUser,
    org_ids: set[UUID],
    dept_pairs: list[tuple[UUID, UUID]],
) -> bool:
    if _is_root_user(current_user):
        # Root should not see tenant/user connectors from org/dept admins or users.
        # Keep root visibility limited to root-owned global connectors.
        return (
            str(getattr(row, "created_by", "")) == str(current_user.id)
            and row.org_id is None
            and row.dept_id is None
        )

    role = normalize_role(str(current_user.role))
    # Super admins can view any connector that belongs to organizations they administer.
    if role == "super_admin" and row.org_id and row.org_id in org_ids:
        return True

    visibility = _normalize_visibility(getattr(row, "visibility", "private"))
    user_id = str(current_user.id)
    dept_id_set = {str(dept_id) for _, dept_id in dept_pairs}

    if visibility == "private":
        return str(row.created_by) == user_id or user_id in set(row.shared_user_ids or [])
    if getattr(row, "public_scope", None) == "organization":
        return bool(row.org_id and row.org_id in org_ids)
    if getattr(row, "public_scope", None) == "department":
        dept_candidates = set(row.public_dept_ids or [])
        if row.dept_id:
            dept_candidates.add(str(row.dept_id))
        return bool(dept_candidates.intersection(dept_id_set))
    return False


# ---------- DB Connection helper ----------

def _test_db_connection(provider: str, host: str, port: int, database_name: str,
                        schema_name: str, username: str, password: str,
                        ssl_enabled: bool) -> dict:
    """Test a database connection and optionally fetch schema metadata."""
    start = time.time()

    if provider == "postgresql":
        import psycopg2
        conn_params = {
            "host": host,
            "port": port,
            "dbname": database_name,
            "user": username,
            "password": password,
            "connect_timeout": 10,
        }
        if ssl_enabled:
            conn_params["sslmode"] = "require"

        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute("SELECT 1")

        # Fetch table/column metadata
        cur.execute("""
            SELECT table_name, column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name NOT LIKE 'pg_%%'
              AND table_name NOT LIKE 'sql_%%'
            ORDER BY table_name, ordinal_position
        """, (schema_name,))
        columns = cur.fetchall()

        tables = {}
        for tbl, col, dtype, nullable, default in columns:
            if tbl not in tables:
                tables[tbl] = {"table_name": tbl, "columns": []}
            tables[tbl]["columns"].append({
                "name": col,
                "type": dtype,
                "nullable": nullable == "YES",
            })

        # Get row counts
        for tbl in tables:
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{schema_name}"."{tbl}"')
                tables[tbl]["row_count"] = cur.fetchone()[0]
            except Exception:
                tables[tbl]["row_count"] = None

        cur.close()
        conn.close()
        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "success": True,
            "message": f"Connected successfully. Found {len(tables)} tables.",
            "latency_ms": latency_ms,
            "tables_metadata": list(tables.values()),
        }
    elif provider == "oracle":
        import oracledb

        dsn = oracledb.makedsn(host, port, service_name=database_name)
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM DUAL")

        # Fetch table/column metadata from Oracle data dictionary
        cur.execute("""
            SELECT table_name, column_name, data_type,
                   CASE WHEN nullable = 'Y' THEN 'YES' ELSE 'NO' END AS is_nullable
            FROM user_tab_columns
            WHERE table_name NOT LIKE 'BIN$%%'
            ORDER BY table_name, column_id
        """)
        columns = cur.fetchall()

        tables = {}
        for tbl, col, dtype, nullable in columns:
            if tbl not in tables:
                tables[tbl] = {"table_name": tbl, "columns": []}
            tables[tbl]["columns"].append({
                "name": col,
                "type": dtype,
                "nullable": nullable == "YES",
            })

        cur.close()
        conn.close()
        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "success": True,
            "message": f"Connected successfully. Found {len(tables)} tables/views.",
            "latency_ms": latency_ms,
            "tables_metadata": list(tables.values()),
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' is not yet supported. Supported: postgresql, oracle",
        )


def _test_azure_blob_connection(config: dict) -> dict:
    """Test an Azure Blob Storage connection."""
    start = time.time()
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="azure-storage-blob not installed. Install with: pip install azure-storage-blob",
        )

    connection_string = config.get("connection_string", "")
    container_name = config.get("container_name", "")

    if not connection_string:
        raise HTTPException(status_code=400, detail="connection_string is required for Azure Blob connector")
    if not container_name:
        raise HTTPException(status_code=400, detail="container_name is required for Azure Blob connector")

    client = BlobServiceClient.from_connection_string(connection_string)
    container_client = client.get_container_client(container_name)
    blobs = list(container_client.list_blobs())
    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "success": True,
        "message": f"Connected successfully. Found {len(blobs)} blobs in '{container_name}'.",
        "latency_ms": latency_ms,
        "tables_metadata": None,
    }


def _test_sharepoint_connection(config: dict) -> dict:
    """Test a SharePoint connection."""
    start = time.time()
    try:
        from office365.runtime.auth.client_credential import ClientCredential
        from office365.sharepoint.client_context import ClientContext
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="Office365-REST-Python-Client not installed. Install with: pip install Office365-REST-Python-Client",
        )

    site_url = config.get("site_url", "")
    client_id = config.get("client_id", "")
    client_secret = config.get("client_secret", "")

    if not site_url or not client_id or not client_secret:
        raise HTTPException(
            status_code=400,
            detail="site_url, client_id, and client_secret are required for SharePoint connector",
        )

    credentials = ClientCredential(client_id, client_secret)
    ctx = ClientContext(site_url).with_credentials(credentials)
    web = ctx.web
    ctx.load(web)
    ctx.execute_query()

    latency_ms = round((time.time() - start) * 1000, 2)
    return {
        "success": True,
        "message": f"Connected successfully to SharePoint site: {web.url}",
        "latency_ms": latency_ms,
        "tables_metadata": None,
    }


# ---------- Endpoints ----------

@router.get("")
@router.get("/")
async def list_connectors(
    current_user: CurrentActiveUser,
    session: DbSession,
) -> list[dict]:
    await _require_connector_permission(current_user, "connectore_page")
    query = select(ConnectorCatalogue).order_by(ConnectorCatalogue.name.asc())
    rows = (await session.exec(query)).all()
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    visible_rows = [row for row in rows if _can_access_connector(row, current_user, org_ids, dept_pairs)]
    return [_serialize_connector(row) for row in visible_rows]


@router.get("/visibility-options")
async def get_connector_visibility_options(
    current_user: CurrentActiveUser,
    session: DbSession,
) -> dict:
    await _require_connector_permission(current_user, "connectore_page")
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    role = normalize_role(str(current_user.role))

    organizations = []
    if role == "root":
        org_rows = (await session.exec(select(Organization.id, Organization.name))).all()
        organizations = [{"id": str(r[0]), "name": r[1]} for r in org_rows]
    elif org_ids:
        org_rows = (
            await session.exec(select(Organization.id, Organization.name).where(Organization.id.in_(list(org_ids))))
        ).all()
        organizations = [{"id": str(r[0]), "name": r[1]} for r in org_rows]

    dept_ids = {dept_id for _, dept_id in dept_pairs}
    departments = []
    if role == "root":
        dept_rows = (await session.exec(select(Department.id, Department.name, Department.org_id))).all()
        departments = [{"id": str(r[0]), "name": r[1], "org_id": str(r[2])} for r in dept_rows]
    elif role == "super_admin" and org_ids:
        dept_rows = (
            await session.exec(
                select(Department.id, Department.name, Department.org_id).where(Department.org_id.in_(list(org_ids)))
            )
        ).all()
        departments = [{"id": str(r[0]), "name": r[1], "org_id": str(r[2])} for r in dept_rows]
    elif dept_ids:
        dept_rows = (
            await session.exec(
                select(Department.id, Department.name, Department.org_id).where(Department.id.in_(list(dept_ids)))
            )
        ).all()
        departments = [{"id": str(r[0]), "name": r[1], "org_id": str(r[2])} for r in dept_rows]

    private_share_users = []
    if role == "department_admin" and dept_ids:
        primary_dept = sorted(dept_ids, key=str)[0]
        user_rows = (
            await session.exec(
                select(User.id, User.email)
                .join(UserDepartmentMembership, UserDepartmentMembership.user_id == User.id)
                .where(
                    UserDepartmentMembership.department_id == primary_dept,
                    UserDepartmentMembership.status == "active",
                    User.email.is_not(None),
                )
            )
        ).all()
        private_share_users = [{"id": str(r[0]), "email": r[1]} for r in user_rows if r[1]]

    return {
        "organizations": organizations,
        "departments": departments,
        "private_share_users": private_share_users,
        "role": role,
    }


@router.post("")
@router.post("/")
async def create_connector(
    payload: ConnectorPayload,
    current_user: CurrentActiveUser,
    session: DbSession,
) -> dict:
    await _require_connector_permission(current_user, "connectore_page")
    await _require_connector_permission(current_user, "add_connector")

    visibility, public_scope, public_dept_ids, shared_user_ids = await _enforce_creation_scope(
        session, current_user, payload
    )
    await _ensure_connector_name_available(session, payload.name, payload.org_id, payload.dept_id)
    now = datetime.now(timezone.utc)
    provider = payload.provider.lower()

    if provider in STORAGE_PROVIDERS:
        # Azure Blob / SharePoint: credentials go into provider_config, not DB fields
        raw_config = payload.provider_config or {}
        encrypted_config = _encrypt_provider_config(provider, raw_config)
        row = ConnectorCatalogue(
            name=payload.name,
            description=payload.description,
            provider=provider,
            host=None,
            port=None,
            database_name=None,
            schema_name=None,
            username=None,
            password_encrypted=None,
            ssl_enabled=False,
            provider_config=encrypted_config,
            status="disconnected",
            is_custom=payload.is_custom,
            org_id=payload.org_id,
            dept_id=payload.dept_id,
            visibility=visibility,
            public_scope=public_scope,
            public_dept_ids=public_dept_ids,
            shared_user_ids=shared_user_ids,
            created_by=current_user.id,
            updated_by=current_user.id,
            created_at=now,
            updated_at=now,
        )
    else:
        # DB providers: use standard DB fields
        row = ConnectorCatalogue(
            name=payload.name,
            description=payload.description,
            provider=provider,
            host=payload.host,
            port=payload.port,
            database_name=payload.database_name,
            schema_name=payload.schema_name,
            username=payload.username,
            password_encrypted=_encrypt_password(payload.password) if payload.password else None,
            ssl_enabled=payload.ssl_enabled,
            provider_config=None,
            status="disconnected",
            is_custom=payload.is_custom,
            org_id=payload.org_id,
            dept_id=payload.dept_id,
            visibility=visibility,
            public_scope=public_scope,
            public_dept_ids=public_dept_ids,
            shared_user_ids=shared_user_ids,
            created_by=current_user.id,
            updated_by=current_user.id,
            created_at=now,
            updated_at=now,
        )

    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _serialize_connector(row)


@router.patch("/{connector_id}")
async def update_connector(
    connector_id: UUID,
    payload: ConnectorUpdatePayload,
    current_user: CurrentActiveUser,
    session: DbSession,
) -> dict:
    await _require_connector_permission(current_user, "connectore_page")
    await _require_connector_permission(current_user, "add_connector")

    row = await session.get(ConnectorCatalogue, connector_id)
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    if not _can_access_connector(row, current_user, org_ids, dept_pairs):
        raise HTTPException(status_code=403, detail="Connector is outside your visibility scope")

    if payload.org_id is None:
        payload.org_id = row.org_id
    if payload.dept_id is None and payload.public_scope != "organization":
        payload.dept_id = row.dept_id
    if payload.visibility is None:
        payload.visibility = row.visibility
    if payload.public_scope is None:
        payload.public_scope = row.public_scope
    if payload.public_dept_ids is None:
        payload.public_dept_ids = [UUID(v) for v in (row.public_dept_ids or [])]

    visibility, public_scope, public_dept_ids, shared_user_ids = await _enforce_creation_scope(
        session, current_user, payload
    )
    if payload.name is not None:
        await _ensure_connector_name_available(
            session,
            payload.name,
            payload.org_id,
            payload.dept_id,
            exclude_id=connector_id,
        )
    now = datetime.now(timezone.utc)

    if payload.name is not None:
        row.name = payload.name
    if payload.description is not None:
        row.description = payload.description
    if payload.provider is not None:
        row.provider = payload.provider.lower()

    effective_provider = row.provider

    if effective_provider in STORAGE_PROVIDERS:
        # Storage provider: update provider_config, clear DB fields
        if payload.provider_config is not None:
            row.provider_config = _encrypt_provider_config(effective_provider, payload.provider_config)
        row.host = None
        row.port = None
        row.database_name = None
        row.schema_name = None
        row.username = None
        row.password_encrypted = None
        row.ssl_enabled = False
    else:
        # DB provider: update DB fields
        if payload.host is not None:
            row.host = payload.host
        if payload.port is not None:
            row.port = payload.port
        if payload.database_name is not None:
            row.database_name = payload.database_name
        if payload.schema_name is not None:
            row.schema_name = payload.schema_name
        if payload.username is not None:
            row.username = payload.username
        if payload.password is not None:
            row.password_encrypted = _encrypt_password(payload.password)
        if payload.ssl_enabled is not None:
            row.ssl_enabled = payload.ssl_enabled
        row.provider_config = None

    if payload.is_custom is not None:
        row.is_custom = payload.is_custom
    row.org_id = payload.org_id
    row.dept_id = payload.dept_id
    row.visibility = visibility
    row.public_scope = public_scope
    row.public_dept_ids = public_dept_ids
    row.shared_user_ids = shared_user_ids
    row.updated_by = current_user.id
    row.updated_at = now

    await session.commit()
    await session.refresh(row)
    return _serialize_connector(row)


@router.delete("/{connector_id}")
async def delete_connector(
    connector_id: UUID,
    current_user: CurrentActiveUser,
    session: DbSession,
) -> dict:
    await _require_connector_permission(current_user, "connectore_page")
    await _require_connector_permission(current_user, "add_connector")

    row = await session.get(ConnectorCatalogue, connector_id)
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    if not _can_access_connector(row, current_user, org_ids, dept_pairs):
        raise HTTPException(status_code=403, detail="Connector is outside your visibility scope")

    await session.delete(row)
    await session.commit()
    return {"message": "Connector deleted successfully"}


@router.post("/{connector_id}/test-connection")
async def test_connector_connection(
    connector_id: UUID,
    current_user: CurrentActiveUser,
    session: DbSession,
    override: TestConnectionPayload | None = None,
) -> dict:
    """Test connectivity to the configured database and refresh schema metadata."""
    row = await session.get(ConnectorCatalogue, connector_id)
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    await _require_connector_permission(current_user, "connectore_page")
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    if not _can_access_connector(row, current_user, org_ids, dept_pairs):
        raise HTTPException(status_code=403, detail="Connector is outside your visibility scope")

    provider = override.provider if override and override.provider else row.provider

    try:
        if provider in STORAGE_PROVIDERS:
            # Use provider_config (override or stored, decrypted)
            if override and override.provider_config:
                config = override.provider_config
            else:
                config = _decrypt_provider_config(provider, row.provider_config or {})

            if provider == "azure_blob":
                result = _test_azure_blob_connection(config)
            else:  # sharepoint
                result = _test_sharepoint_connection(config)
        else:
            # DB provider
            host = override.host if override and override.host else row.host
            port = override.port if override and override.port else row.port
            database_name = override.database_name if override and override.database_name else row.database_name
            schema_name = override.schema_name if override and override.schema_name else row.schema_name
            username = override.username if override and override.username else row.username
            password = (
                override.password if override and override.password
                else (_decrypt_password(row.password_encrypted) if row.password_encrypted else "")
            )
            ssl_enabled = override.ssl_enabled if override and override.ssl_enabled is not None else row.ssl_enabled
            result = _test_db_connection(provider, host, port, database_name, schema_name, username, password, ssl_enabled)

        now = datetime.now(timezone.utc)
        row.status = "connected"
        row.tables_metadata = result.get("tables_metadata")
        row.last_tested_at = now
        row.updated_at = now
        await session.commit()
        await session.refresh(row)
        return result
    except HTTPException:
        raise
    except Exception as e:
        now = datetime.now(timezone.utc)
        row.status = "error"
        row.last_tested_at = now
        row.updated_at = now
        await session.commit()
        return {
            "success": False,
            "message": f"Connection failed: {e!s}",
            "latency_ms": None,
            "tables_metadata": None,
        }


@router.post("/test-connection")
async def test_connector_connection_payload(
    payload: TestConnectionPayload,
    current_user: CurrentActiveUser,
) -> dict:
    """Test connectivity from unsaved connector payload (used by create modal)."""
    await _require_connector_permission(current_user, "connectore_page")
    await _require_connector_permission(current_user, "add_connector")

    provider = (payload.provider or "").strip().lower()
    if not provider:
        raise HTTPException(status_code=400, detail="provider is required")

    try:
        if provider in STORAGE_PROVIDERS:
            config = payload.provider_config or {}
            if provider == "azure_blob":
                return _test_azure_blob_connection(config)
            return _test_sharepoint_connection(config)

        if not payload.host or not payload.port or not payload.database_name or not payload.username:
            raise HTTPException(
                status_code=400,
                detail="host, port, database_name, username are required for DB providers",
            )
        return _test_db_connection(
            provider=provider,
            host=payload.host,
            port=payload.port,
            database_name=payload.database_name,
            schema_name=payload.schema_name or "public",
            username=payload.username,
            password=payload.password or "",
            ssl_enabled=bool(payload.ssl_enabled),
        )
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "message": f"Connection failed: {e!s}",
            "latency_ms": None,
            "tables_metadata": None,
        }


@router.post("/{connector_id}/disconnect")
async def disconnect_connector(
    connector_id: UUID,
    current_user: CurrentActiveUser,
    session: DbSession,
) -> dict:
    """Manually disconnect a connector (set status to 'disconnected')."""
    await _require_connector_permission(current_user, "connectore_page")
    await _require_connector_permission(current_user, "add_connector")

    row = await session.get(ConnectorCatalogue, connector_id)
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    if not _can_access_connector(row, current_user, org_ids, dept_pairs):
        raise HTTPException(status_code=403, detail="Connector is outside your visibility scope")

    now = datetime.now(timezone.utc)
    row.status = "disconnected"
    row.updated_at = now
    row.updated_by = current_user.id
    await session.commit()
    await session.refresh(row)
    return {"message": "Connector disconnected", "status": "disconnected"}


@router.get("/{connector_id}/schema")
async def get_connector_schema(
    connector_id: UUID,
    current_user: CurrentActiveUser,
    session: DbSession,
) -> dict:
    """Return cached schema metadata for a connector."""
    await _require_connector_permission(current_user, "connectore_page")
    row = await session.get(ConnectorCatalogue, connector_id)
    if not row:
        raise HTTPException(status_code=404, detail="Connector not found")
    org_ids, dept_pairs = await _get_scope_memberships(session, current_user.id)
    if not _can_access_connector(row, current_user, org_ids, dept_pairs):
        raise HTTPException(status_code=403, detail="Connector is outside your visibility scope")

    return {
        "connector_id": str(row.id),
        "connector_name": row.name,
        "provider": row.provider,
        "database_name": row.database_name,
        "schema_name": row.schema_name,
        "status": row.status,
        "tables_metadata": row.tables_metadata or [],
        "last_tested_at": row.last_tested_at.isoformat() if row.last_tested_at else None,
    }

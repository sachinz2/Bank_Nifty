import os
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

DB_USER = os.getenv("BN_DB_USER", "root")
DB_PASS = os.getenv("BN_DB_PASS", "sachin")
DB_HOST = os.getenv("BN_DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("BN_DB_PORT", "3306")
DB_NAME = os.getenv("BN_DB_NAME", "bank_nifty")

def _make_url(include_db: bool = True) -> str:
    db_part = f"/{DB_NAME}" if include_db else "/"
    return f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}{db_part}?charset=utf8mb4"

def make_engine(echo: bool = False) -> Engine:
    return create_engine(_make_url(include_db=True), echo=echo, pool_pre_ping=True)

def _make_server_engine(echo: bool = False) -> Engine:
    return create_engine(_make_url(include_db=False), echo=echo, pool_pre_ping=True)

def init_schema(setup_sql_path: Optional[str] = None) -> None:
    if setup_sql_path is None:
        setup_sql_path = os.path.join(os.path.dirname(__file__), "db", "setup.sql")
    if not os.path.exists(setup_sql_path):
        raise FileNotFoundError(f"setup SQL not found: {setup_sql_path}")
    with open(setup_sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    engine = _make_server_engine()
    try:
        with engine.connect() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
            conn.commit()
    except SQLAlchemyError as ex:
        raise RuntimeError(f"Failed to initialize DB schema: {ex}") from ex

def insert_market_data(engine: Engine, symbol: str, ts, o: Optional[float] = None, h: Optional[float] = None,
                       l: Optional[float] = None, c: Optional[float] = None, volume: Optional[int] = None,
                       raw_json: Optional[str] = None) -> int:
    sql = text("""
        INSERT INTO market_data (symbol, ts, `open`, `high`, `low`, `close`, volume, raw_json)
        VALUES (:symbol, :ts, :o, :h, :l, :c, :volume, :raw_json)
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "symbol": symbol,
                "ts": ts,
                "o": o, "h": h, "l": l, "c": c,
                "volume": volume,
                "raw_json": raw_json
            })
            last_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
            return int(last_id) if last_id is not None else 0
    except SQLAlchemyError:
        raise

def get_latest_market(engine: Engine, symbol: str, limit: int = 1) -> List[Dict[str, Any]]:
    sql = text("SELECT * FROM market_data WHERE symbol = :symbol ORDER BY ts DESC LIMIT :limit")
    with engine.connect() as conn:
        rows = conn.execute(sql, {"symbol": symbol, "limit": limit}).mappings().all()
        return [dict(r) for r in rows]

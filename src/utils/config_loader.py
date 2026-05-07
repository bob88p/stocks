import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class JobMeta:
    name: str
    version: str
    description: str


@dataclass
class InputConfig:
    source_name: str
    input_type: str
    symbols: List[str]
    period: str
    interval: str
    auto_adjust: bool
    has_header: bool
    input_schema: dict


@dataclass
class RejectionConfig:
    rejection_path: str
    rejection_type: str
    max_rejection_rate: float


@dataclass
class LayerTarget:
    schema: str
    table: str


@dataclass
class OutputConfig:
    output_type: str
    save_mode: str
    partition_cols: List[str] = field(default_factory=list)
    bronze: Optional[LayerTarget] = None
    silver: Optional[LayerTarget] = None
    gold: Optional[LayerTarget] = None


@dataclass
class ETLConfig:
    rules: List[dict] = field(default_factory=list)


@dataclass
class QualityConfig:
    checks: List[str] = field(default_factory=list)


# ============================================================
# DB CONFIG
# ============================================================

@dataclass
class DBConfig:
    server: str
    port: int
    database: str
    username: str
    password: str
    driver: str
    trusted_connection: bool

    def get_sqlalchemy_url(self) -> str:
        """
        SQL Server connection using pyodbc.
        Omits port for localhost to allow Shared Memory fallback if TCP is disabled.
        """
        driver = self.driver.replace(" ", "+")
        
        server_part = self.server
        if self.server.lower() in ("localhost", "127.0.0.1", ".") and self.port == 1433:
            server_part = self.server
        else:
            server_part = f"{self.server}:{self.port}"

        return (
            f"mssql+pyodbc://{self.username}:{self.password}"
            f"@{server_part}/{self.database}"
            f"?driver={driver}"
        )

    def get_pyodbc_string(self) -> str:
        if self.trusted_connection:
            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )

        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
        )


@dataclass
class JobConfig:
    job: JobMeta
    input: InputConfig
    rejection: RejectionConfig
    output: OutputConfig
    etl: ETLConfig
    quality: QualityConfig
    db: DBConfig


# ============================================================
# ENV LOADER (SAFE)
# ============================================================

def _load_db_from_env() -> DBConfig:
    required = ["DB_SERVER", "DB_NAME", "DB_USER", "DB_PASSWORD"]

    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

    return DBConfig(
        server=os.environ["DB_SERVER"],
        port=int(os.getenv("DB_PORT", 1433)),
        database=os.environ["DB_NAME"],
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        driver=os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
        trusted_connection=os.getenv("DB_TRUSTED_CONNECTION", "no").lower() == "yes",
    )


# ============================================================
# OUTPUT PARSER (SAFE)
# ============================================================

def _parse_output(raw: dict) -> OutputConfig:
    layers = raw.get("layers", {})

    return OutputConfig(
        output_type=raw["output_type"],
        save_mode=raw["save_mode"],
        partition_cols=raw.get("partition_cols", []),
        bronze=LayerTarget(**layers["bronze"]) if "bronze" in layers else None,
        silver=LayerTarget(**layers["silver"]) if "silver" in layers else None,
        gold=LayerTarget(**layers["gold"]) if "gold" in layers else None,
    )


# ============================================================
# MAIN LOADER
# ============================================================

def load_config(path: str) -> JobConfig:

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return JobConfig(
        job=JobMeta(**raw["job"]),
        input=InputConfig(**raw["input"]),
        rejection=RejectionConfig(**raw["rejection"]),
        output=_parse_output(raw["output"]),
        etl=ETLConfig(rules=raw.get("etl", {}).get("rules", [])),
        quality=QualityConfig(checks=raw.get("quality", {}).get("checks", [])),
        db=_load_db_from_env(),
    )
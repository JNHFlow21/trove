#!/usr/bin/env python3
"""Generate local-only redacted real-vault E2E cases.

The output is intended for the user's runtime Vault proof area, not Git. It
contains query strings and safe metadata/oracles only; it never stores message
bodies, snippets, tokens, source paths, or contact details.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    case_id: str
    category: str
    query: str
    min_results: int = 1
    oracle: str = "min_results"
    required: bool = False
    expected_paths_any: tuple[str, ...] = ()
    context: bool = True


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def sqlite_path(vault: Path) -> Path:
    return vault / "index" / "trove.sqlite"


def connect(vault: Path) -> sqlite3.Connection:
    db = sqlite_path(vault)
    if not db.exists():
        raise SystemExit(f"missing Vault sqlite index: {db}")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_proof_output(path: Path, vault: Path) -> Path:
    resolved = path.expanduser().resolve()
    repo_root = ROOT.resolve()
    proof_root = (vault / "proof").resolve()
    if resolved == repo_root or is_relative_to(resolved, repo_root):
        raise SystemExit("real-vault proof output must not be written inside the source repo")
    if not is_relative_to(resolved, proof_root):
        raise SystemExit("real-vault proof output must stay under the selected Vault proof directory")
    return resolved


def count_table(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def date_range(conn: sqlite3.Connection) -> dict[str, str | None]:
    row = conn.execute("SELECT MIN(timestamp) min_ts, MAX(timestamp) max_ts FROM messages").fetchone()
    return {"min": row["min_ts"], "max": row["max_ts"]}


def terms_for_sql(query: str) -> list[str]:
    if " " in query:
        return [t for t in query.split() if len(t) >= 2]
    return [query]


def sql_hits(conn: sqlite3.Connection, query: str, limit: int = 8) -> list[sqlite3.Row]:
    terms = terms_for_sql(query)
    clauses = []
    params: list[str] = []
    for term in terms:
        clauses.append("(content LIKE ? OR conversation_title LIKE ? OR sender_name LIKE ?)")
        params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
    where = " AND ".join(clauses) if clauses else "1=1"
    return list(conn.execute(
        f"""
        SELECT citation, account_id, conversation_id, conversation_type, timestamp
        FROM messages
        WHERE {where}
        ORDER BY timestamp
        LIMIT ?
        """,
        (*params, limit),
    ))


def sql_count(conn: sqlite3.Connection, query: str) -> int:
    terms = terms_for_sql(query)
    clauses = []
    params: list[str] = []
    for term in terms:
        clauses.append("(content LIKE ? OR conversation_title LIKE ? OR sender_name LIKE ?)")
        params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
    where = " AND ".join(clauses) if clauses else "1=1"
    return int(conn.execute(f"SELECT COUNT(*) FROM messages WHERE {where}", params).fetchone()[0])


def conversation_shape(conn: sqlite3.Connection, conversation_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT conversation_id, conversation_type, COUNT(*) message_count,
               MIN(timestamp) min_ts, MAX(timestamp) max_ts
        FROM messages
        WHERE conversation_id=?
        GROUP BY conversation_id, conversation_type
        """,
        (conversation_id,),
    ).fetchone()
    if not row:
        return {}
    return {
        "conversation_id": row["conversation_id"],
        "conversation_type": row["conversation_type"],
        "message_count": row["message_count"],
        "min_ts": row["min_ts"],
        "max_ts": row["max_ts"],
    }


PRODUCT_IMPORTANT_TERMS = ['客户', '卡', '价格', '预算', '审批', '试点', '团队', '决定', '上线', 'token', 'evidence', 'citation', 'context', 'vault']


def product_uses_all_query_terms(query: str) -> bool:
    q = query.lower()
    dictionary_terms = [term for term in PRODUCT_IMPORTANT_TERMS if term in q]
    if dictionary_terms:
        return False
    return " " in query


def default_candidates() -> list[Candidate]:
    return [
        Candidate("hf-model", "high_frequency", "模型", min_results=5, expected_paths_any=("exact", "fts")),
        Candidate("hf-ci", "high_frequency", "CI", min_results=5, expected_paths_any=("exact", "fts")),
        Candidate("hf-project", "high_frequency", "项目", min_results=5, expected_paths_any=("exact", "fts")),
        Candidate("sparse-github-ci", "sparse_precision", "GitHub CI", expected_paths_any=("exact", "fts")),
        Candidate("sparse-ci-failure", "sparse_precision", "CI 失败", expected_paths_any=("exact", "fts")),
        Candidate("sparse-local-model", "sparse_precision", "本地 模型", expected_paths_any=("exact", "fts")),
        Candidate("sparse-privacy-permission", "sparse_precision", "权限 隐私", expected_paths_any=("exact", "fts")),
        Candidate("sparse-douyin-video", "sparse_precision", "抖音 视频", expected_paths_any=("exact", "fts")),
        Candidate("sparse-price-quote", "sparse_precision", "价格 报价", expected_paths_any=("exact", "fts")),
        Candidate("dontbesilent-course-enroll", "real_scenario", "dontbesilent 线下课 报名", required=True, expected_paths_any=("exact", "fts")),
        Candidate("dontbesilent-course-logistics", "real_scenario", "下单 时间地点", expected_paths_any=("exact", "fts")),
        Candidate("semantic-followups", "semantic", "最近需要跟进的事情有哪些", oracle="semantic_min_results", expected_paths_any=("vector",)),
        Candidate("semantic-local-model", "semantic", "找一下和本地模型部署有关的讨论", oracle="semantic_min_results", expected_paths_any=("vector",)),
        Candidate("semantic-content", "semantic", "找一下短视频内容生产相关的聊天", oracle="semantic_min_results", expected_paths_any=("vector",)),
    ]


def build_cases(vault: Path, *, max_cases: int) -> dict[str, Any]:
    conn = connect(vault)
    try:
        counts = {
            "accounts": count_table(conn, "accounts"),
            "conversations": count_table(conn, "conversations"),
            "messages": count_table(conn, "messages"),
            "chunks": count_table(conn, "message_fts"),
        }
        if counts["messages"] <= 0:
            raise SystemExit("real Vault has no messages")
        cases: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for candidate in default_candidates():
            hit_count = sql_count(conn, candidate.query) if candidate.category != "semantic" else None
            hits = sql_hits(conn, candidate.query, limit=8) if candidate.category != "semantic" else []
            if candidate.category != "semantic" and (hit_count or 0) < candidate.min_results:
                item = {"case_id": candidate.case_id, "query": candidate.query, "hit_count": hit_count, "reason": "insufficient_sql_hits"}
                if candidate.required:
                    raise SystemExit(f"required case unavailable: {json.dumps(item, ensure_ascii=False)}")
                skipped.append(item)
                continue
            oracle: dict[str, Any] = {
                "type": candidate.oracle,
                "min_results": candidate.min_results,
                "expected_retrieval_paths_any": list(candidate.expected_paths_any),
            }
            safe_hits = []
            for row in hits[:3]:
                safe_hits.append({
                    "citation": row["citation"],
                    "conversation_id": row["conversation_id"],
                    "conversation_hash": stable_hash(row["conversation_id"]),
                    "conversation_type": row["conversation_type"],
                    "timestamp": row["timestamp"],
                })
            if safe_hits and (candidate.required or candidate.category == "real_scenario"):
                oracle["expected_any_citation"] = [h["citation"] for h in safe_hits]
                oracle["expected_any_conversation_id"] = sorted({h["conversation_id"] for h in safe_hits})
            cases.append({
                "case_id": candidate.case_id,
                "category": candidate.category,
                "query": candidate.query,
                "limit": 5,
                "context": candidate.context,
                "source": "real_vault_index",
                "discovered_hit_count": hit_count,
                "safe_hits": safe_hits,
                "oracle": oracle,
            })
            if len(cases) >= max_cases:
                break
        return {
            "schema_version": 1,
            "created_at": now_iso(),
            "privacy": {
                "raw_message_bodies_included": False,
                "snippets_included": False,
                "token_values_included": False,
                "private_paths_included": False,
            },
            "vault": {
                "label": "configured-vault",
                "counts": counts,
                "date_range": date_range(conn),
            },
            "cases": cases,
            "skipped_candidates": skipped,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cases", type=int, default=14)
    args = parser.parse_args()
    vault = Path(args.vault).expanduser().resolve()
    data = build_cases(vault, max_cases=args.max_cases)
    out = validate_proof_output(Path(args.out), vault)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "written": out.name,
        "cases": len(data["cases"]),
        "messages": data["vault"]["counts"]["messages"],
        "raw_message_bodies_included": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

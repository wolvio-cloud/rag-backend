"""Apply uniform PDF columns to Supabase. Requires SUPABASE_DB_URL in backend/.env

Example SUPABASE_DB_URL:
postgresql://postgres.[project-ref]:[YOUR-DB-PASSWORD]@aws-0-ap-south-1.pooler.supabase.com:6543/postgres

Find password in: Supabase Dashboard → Project Settings → Database
"""

import sys
from pathlib import Path

MIGRATION_SQL = """
alter table public.documents
  add column if not exists uniform_pdf_url text,
  add column if not exists uniform_pdf_name text;
"""


def main() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Install python-dotenv first.")
        return 1

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    db_url = __import__("os").environ.get("SUPABASE_DB_URL", "").strip()

    if not db_url:
        print("SUPABASE_DB_URL is not set in backend/.env")
        print("\nRun this SQL manually in Supabase SQL Editor instead:\n")
        print(MIGRATION_SQL)
        print("\nhttps://supabase.com/dashboard/project/vkodyxmxvjqoyjllslhh/sql/new")
        return 1

    try:
        import psycopg2
    except ImportError:
        print("Install psycopg2-binary: pip install psycopg2-binary")
        return 1

    try:
        with psycopg2.connect(db_url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(MIGRATION_SQL)
        print("Migration applied successfully.")
        return 0
    except Exception as exc:
        print(f"Migration failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

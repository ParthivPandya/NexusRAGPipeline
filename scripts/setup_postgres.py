import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus.config import get_config
from nexus.storage.postgres_store import PostgresStore

def main():
    print("Setting up PostgreSQL...")
    config = get_config()
    store = PostgresStore(dsn=config.postgres_dsn)
    store.setup()
    store.close()
    print("PostgreSQL setup complete.")

if __name__ == "__main__":
    main()

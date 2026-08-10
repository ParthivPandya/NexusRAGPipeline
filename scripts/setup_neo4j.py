import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus.config import get_config
from nexus.storage.neo4j_store import Neo4jStore

def main():
    print("Setting up Neo4j...")
    config = get_config()
    store = Neo4jStore(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
    )
    store.setup()
    store.close()
    print("Neo4j setup complete.")

if __name__ == "__main__":
    main()

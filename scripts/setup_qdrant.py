import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus.config import get_config
from nexus.storage.qdrant_store import QdrantStore

def main():
    print("Setting up Qdrant...")
    config = get_config()
    store = QdrantStore(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection=config.qdrant_collection,
        dim=config.embedding_dim,
    )
    store.setup()
    print("Qdrant setup complete.")

if __name__ == "__main__":
    main()

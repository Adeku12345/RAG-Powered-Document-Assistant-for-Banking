"""Central configuration. Change settings here, not in the other modules."""
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")  # loads ANTHROPIC_API_KEY

# Paths
PDF_DIR = ROOT_DIR / "data" / "pdf"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"
EVAL_SET_PATH = ROOT_DIR / "data" / "eval_set.json"
COLLECTION_NAME = "banking_docs"

# Models (the embedding model must be the same at ingest time and query time)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-haiku-5-5"
LLM_MAX_TOKENS = 1000

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 150  # drops title-only chunks

# Retrieval
TOP_K = 4

# Repeated header/footer lines to strip from every page (regex, matched per line)
NOISE_PATTERNS = [
    r"^.*Harborview Bank \(fictional\).*$",
    r"^\s*(\*\*)?Page \d+(\*\*)?\s*$",
]

import sys

from dotenv import load_dotenv

# Load `.env` for runtime processes (server, CLI). Skipped under pytest so the
# suite stays hermetic and never inherits a developer's local `.env`.
if "pytest" not in sys.modules:
    load_dotenv()

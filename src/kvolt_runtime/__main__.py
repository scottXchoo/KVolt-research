"""Enable ``python -m kvolt_runtime`` without importing GPU dependencies."""
from .cli import main

raise SystemExit(main())

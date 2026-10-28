"""Default entry point: run the complete offline anchor-selection pipeline."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falhv2.anchor_planners.pipeline import main, plan

if __name__ == "__main__":
    raise SystemExit(main())

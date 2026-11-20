"""GenD v2 experiments: paired FF++ protocol, official GenD baseline, and four candidate extensions."""
import sys

from .paths import GEND_ROOT  # official repo, pinned to commit 387a422

if GEND_ROOT not in sys.path:
    sys.path.insert(0, GEND_ROOT)

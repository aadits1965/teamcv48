"""All filesystem locations in one place. Defaults are the Google Colab + Google Drive layout used for every
reported run; override with environment variables to run elsewhere (behaviour is otherwise identical)."""
import os
from pathlib import Path

GEND_ROOT = os.environ.get("GEND_OFFICIAL_DIR", "/content/GenD")          # official GenD repo @ 387a422
V2_DIR = Path(os.environ.get("GEND_V2_DIR", "/content/drive/MyDrive/GenD_PCR_Project/v2"))  # persistent store
WORK_DIR = Path(os.environ.get("GEND_WORK_DIR", "/content/data"))       # fast local scratch (raw videos, crops)

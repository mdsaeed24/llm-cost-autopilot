"""Entry point for Streamlit Community Cloud (public, read-only demo).

    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # no package install needed on the cloud

from autopilot.dashboard.showcase import main  # noqa: E402

main()

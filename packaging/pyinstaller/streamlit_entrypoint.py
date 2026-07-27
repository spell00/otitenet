import os
import sys
from pathlib import Path

import streamlit.web.cli as stcli


def bundled_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def resolve_path(path: str) -> str:
    return str(bundled_root() / path)


def choose_port(default: int = 8501) -> int:
    return int(os.environ.get("OTITENET_STREAMLIT_PORT", default))


if __name__ == "__main__":
    os.chdir(bundled_root())
    src_path = bundled_root() / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    port = choose_port()
    os.environ["OTITENET_STREAMLIT_PORT"] = str(port)
    app_file = os.environ.get("OTITENET_STREAMLIT_APP", "app_offline.py")

    sys.argv = [
        "streamlit",
        "run",
        resolve_path(app_file),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        "--server.maxUploadSize=200",
    ]
    sys.exit(stcli.main())

# /home/simon/otitenet/otitenet/app/context.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AppContext:
    """
    Shared application context passed to refactored page modules.

    This replaces the old pattern where each tab in app.py directly accessed
    global variables such as conn, cursor, args, device, data_dir, and
    current_model_key.
    """

    conn: Any
    cursor: Any
    args: Any
    is_admin: bool
    data_dir: str
    device: Any

    production_model: Optional[Any] = None
    current_model_key: Optional[Any] = None
    selected_person_id: Optional[int] = None
    user_id: Optional[int] = None
    user_email: Optional[str] = None
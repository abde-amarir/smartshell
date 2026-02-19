import sqlite3
import os
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────────
DP_PATH = Path(__file__).resolve().parent.parent / "data" / "logs" / "commands.db"

# ── Database Setup ──────────────────────────────────────────────────────────────
def init_db():
    """Create the database and commands table if they not exist."""
    DP_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DP_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS commands(
                   id               INTEGER PRIMARY KEY AUTOINCREMENT,
                   command          TEXT        NOT NULL,
                   directory        TEXT        NOT NULL,
                   exit_code       INTEGER     NOT NULL,
                   timestamp        TEXT        NOT NULL,
                   hour             INTEGER     NOT NULL,
                   day_of_week      INTEGER     NOT NULL,
                   session_id       TEXT        NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# ── Core Logger ────────────────────────────────────────────────────────────────
def log_command(command: str, directory: str, exit_code: int, session_id: str):
    """
    Write a single command entry to the database.

    Args:
        command:    The raw command string the user executed.
        directory:  The working directory at time of execution (pwd).
        exit_code:  The exit code of the command (0 = success).
        session_id: A unique ID for the current terminal session.
    """
    # Skip empty or whitespace-only commands
    if not command or not command.strip():
        return
    
    # Skip commands we never want to log or suggest
    BLACKLIST = {
        "rm", "dd", "mkfs", "shutdown", "reboot",
        "poweroff", "halt", "mkswap", "fdisk", "parted"
    }
    base_command = command.strip().split()[0].lstrip("sudo").strip()
    if base_command in BLACKLIST:
        return
    
    now = datetime.now()

    conn = sqlite3.connect(DP_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO commands (command, directory, exit_code, timestamp, hour, day_of_week, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        command.strip(),
        directory,
        exit_code,
        now.isoformat(),
        now.hour,
        now.weekday(),   # 0 = Monday, 6 = Sunday
        session_id
    ))

    conn.commit()
    conn.close()


# ── Entry Point (called from Bash) ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Bash will call this script with 3 arguments:
    # python logger.py "<command>" "<directory>" "<exit_code>" "<session_id>"
    if len(sys.argv) != 5:
        sys.exit(1)

    _, cmd, cwd, code, sid = sys.argv

    init_db()
    log_command(
        command=cmd,
        directory=cwd,
        exit_code=int(code),
        session_id=sid
    )
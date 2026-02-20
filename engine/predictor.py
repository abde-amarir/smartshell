import sqlite3
from pathlib import Path
from models.markov import MarkovPredictor
from processor.normalizer import normalize

DB_PATH    = Path(__file__).resolve().parent.parent / "data" / "logs" / "commands.db"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "markov_model.pkl"


def get_last_commands(n: int = 2, session_id: str = None) -> list[str]:
    """
    Fetch the last N normalized commands from the database.

    Args:
        n:          How many recent commands to retrieve.
        session_id: If provided, restrict to the current session only.

    Returns:
        List of normalized command strings, oldest first.
        e.g. ["git add", "git commit"]
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if session_id:
        cursor.execute("""
            SELECT command FROM commands
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, n))
    else:
        cursor.execute("""
            SELECT command FROM commands
            ORDER BY id DESC
            LIMIT ?
        """, (n,))

    rows = cursor.fetchall()
    conn.close()

    # Reverse so oldest is first, normalize each command
    raw_commands = [row[0] for row in reversed(rows)]
    normalized = [normalize(cmd) for cmd in raw_commands]

    # Filter out any None values from normalization
    return [cmd for cmd in normalized if cmd is not None]


def get_prediction(session_id: str = None) -> dict | None:
    """
    Load the trained model and return the top prediction
    based on the last command in the database.

    Returns:
        dict with keys: "suggested", "probability", "last_command"
        or None if no prediction is available.
    """
    if not MODEL_PATH.exists():
        return None

    if not DB_PATH.exists():
        return None

    model = MarkovPredictor.load(MODEL_PATH)
    last_commands = get_last_commands(n=model.order, session_id=session_id)

    if not last_commands:
        return None

    results = model.predict_from_last(last_commands)

    if not results:
        return None

    top = results[0]

    return {
        "suggested":    top["command"],
        "probability":  top["probability"],
        "last_command": last_commands[-1],
    }
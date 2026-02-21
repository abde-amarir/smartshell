import sqlite3
import pandas as pd
from pathlib import Path
from processor.normalizer import normalize, normalize_directory

DB_PATH     = Path(__file__).resolve().parent.parent / "data" / "logs" / "commands.db"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"

SEQUENCE_LENGTH = 3  # how many previous commands to use as context


def load_raw(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load all commands from SQLite into a DataFrame, ordered by time."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT id, command, directory, exit_code, timestamp, hour, day_of_week, session_id
        FROM commands
        ORDER BY id ASC  
    """, conn)
    conn.close()
    return df

def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalize() to each command.
    Rows that normalize to None are dropped (blacklisted or empty).
    """
    df = df.copy()
    df["command_norm"] = df["command"].apply(normalize)
    df["directory_norm"] = df["directory"].apply(normalize_directory)

    # Drop excluded commands
    df = df[df["command_norm"].notna()].reset_index(drop=True)
    return df

def build_sequence_pairs(df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH) -> pd.DataFrame:
    """
    Build training pairs from the command sequence.

    Each row in the output represents:
        [prev_1, prev_2, prev_3, context features] → target

    Sequences are built WITHIN each session — we never let context
    bleed across separate terminal sessions.

    Args:
        df:              Normalized command DataFrame.
        sequence_length: Number of previous commands to use as context.

    Returns:
        DataFrame with one training sample per row.
    """
    records = []

    for session_id, group in df.groupby("session_id"):
        group = group.reset_index(drop=True)

        # Need at least sequence_length + 1 commands to build one pair
        if len(group) < sequence_length + 1:
            continue

        for i in range(sequence_length, len(group)):
            target_row = group.iloc[i]
            context_rows = group.iloc[i - sequence_length: i]

            record = {
                "target": target_row["command_norm"],
                "hour": target_row["hour"],
                "day_of_week": target_row["day_of_week"],
                "directory": target_row["directory_norm"],
                "session_id": session_id,
            }

            record["prev_exit_code"] = group.iloc[i-1]["exit_code"]
            record["position"] = i / len(group)
            record["session_length"] = len(group)

            # Add previous commands as prev_1, prev_2, prev_3
            # prev_1 is the most recent, prev_N is the oldest
            for j, (_, ctx_row) in enumerate(context_rows.iloc[::-1].iterrows()):
                record[f"prev_{j+1}"] = ctx_row["command_norm"]

            records.append(record)

    return pd.DataFrame(records)

def run_pipeline() -> pd.DataFrame:
    """
    Execute the full pipeline end-to-end.
    Loads raw data → normalizes → builds training pairs → saves to CSV.
    """
    print("Loading raw data...")
    df_raw = load_raw()
    print(f"  {len(df_raw)} raw commands loaded.")

    print("Applying normalization...")
    df_norm = apply_normalization(df_raw)
    print(f"  {len(df_norm)} commands after filtering.")

    print("Building sequence pairs...")
    df_pairs = build_sequence_pairs(df_norm)
    print(f"  {len(df_pairs)} training pairs built.")

    # Save outputs
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(OUTPUT_PATH / "commands_normalized.csv", index=False)
    df_pairs.to_csv(OUTPUT_PATH / "training_pairs.csv", index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    return df_pairs


if __name__ == "__main__":
    df = run_pipeline()
    print("\nSample training pairs:")
    print(df.head(10).to_string())
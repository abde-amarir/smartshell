import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "markov_model.pkl"

def _default_counter():
    return defaultdict(int)

class MarkovPredictor:
    """
    An N-gram Markov Chain command predictor.

    order=1 → uses last 1 command as context
    order=2 → uses last 2 commands as context (more accurate, needs more data)

    Internal structure:
        self.transitions = {
            context_tuple: {next_command: count, ...},
            ...
        }
    Example (order=1):
        {
            ("git add",):   {"git commit": 12, "git status": 3},
            ("git commit",): {"git push": 10, "git log": 2},
        }
    """

    def __init__(self, order: int = 1):
        if order not in (1, 2):
            raise ValueError("Order must be 1 or 2.")
        self.order = order
        self.transitions: dict = defaultdict(_default_counter)
        self.trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> None:
        """
        Build transition counts from a training pairs DataFrame.

        Args:
            df: Output of processor.pipeline.build_sequence_pairs()
                Must contain columns: target, prev_1, prev_2
        """
        required = {"target", "prev_1"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        for _, row in df.iterrows():
            target = row["target"]

            if self.order == 1:
                context = (row["prev_1"],)
            else:
                # order=2 — need prev_2 as well
                if pd.isna(row.get("prev_2")):
                    continue
                context = (row["prev_2"], row["prev_1"])

            self.transitions[context][target] += 1

        self.trained = True
        print(f"Markov model (order={self.order}) trained.")
        print(f"  {len(self.transitions)} unique contexts learned.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, context: tuple, top_n: int = 3) -> list[dict]:
        """
        Predict the most likely next commands given a context tuple.

        Args:
            context: Tuple of the last N commands.
                     order=1 → ("git add",)
                     order=2 → ("git status", "git add")
            top_n:   How many predictions to return.

        Returns:
            List of dicts sorted by probability descending:
            [{"command": "git commit", "probability": 0.80}, ...]
        """
        if not self.trained:
            raise TimeoutError("Model has not been trained yet. Call train() first.")

        if context not in self.transitions:
            return []   # unseen context — no prediction
        
        counts = self.transitions[context]
        total = sum(counts.values())

        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {"command": cmd, "probability": round(count / total, 3)}
            for cmd, count in ranked
        ]
    
    def predict_from_last(self, last_commands: list[str], top_n: int = 3) -> list[dict]:
        """
        Convenience method — pass a plain list of recent commands.

        Args:
            last_commands: Most recent commands, newest last.
                           e.g. ["git status", "git add"] for order=2
        """
        if len(last_commands) < self.order:
            return []
        
        context = tuple(last_commands[-self.order:])
        return self.predict(context, top_n)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        """Save the trained model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "MarkovPredictor":
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded ← {path}")
        return model
    
    # ── Inspection ────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a readable summary of what the model has learned."""
        if not self.trained:
            print("Model not trained yet.")
            return

        print(f"\nMarkov Chain (order={self.order}) Summary")
        print(f"{'─' * 40}")
        print(f"Unique contexts : {len(self.transitions)}")

        total_transitions = sum(
            sum(targets.values())
            for targets in self.transitions.values()
        )
        print(f"Total transitions: {total_transitions}")
        print(f"\nTop learned transitions:")

        # Sort contexts by total count descending
        sorted_contexts = sorted(
            self.transitions.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )[:10]

        for context, targets in sorted_contexts:
            top_target = max(targets, key=targets.get)
            total = sum(targets.values())
            prob = targets[top_target] / total
            print(f"  {' → '.join(context)} → {top_target}  ({prob:.0%}, seen {total}x)")
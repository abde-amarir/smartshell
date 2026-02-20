import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"


class SklearnPredictor:
    """
    Classical ML command predictor using Logistic Regression or Random Forest.

    Features used per training sample:
        - prev_1       : most recent command (categorical)
        - prev_2       : second most recent command (categorical)
        - prev_3       : third most recent command (categorical)
        - hour         : hour of day 0-23 (numeric)
        - day_of_week  : 0=Monday, 6=Sunday (numeric)
        - directory    : normalized working directory (categorical)

    Target:
        - target       : next command to predict
    """

    FEATURE_COLS = ["prev_1", "prev_2", "prev_3", "hour", "day_of_week", "directory"]
    CATEGORICAL  = ["prev_1", "prev_2", "prev_3", "directory"]
    NUMERIC      = ["hour", "day_of_week"]

    def __init__(self, model_type: str = "logreg"):
        """
        Args:
            model_type: "logreg" for Logistic Regression,
                        "rf" for Random Forest.
        """
        if model_type not in ("logreg", "rf"):
            raise ValueError("model_type must be 'logreg' or 'rf'")

        self.model_type = model_type
        self.label_encoder = LabelEncoder()
        self.pipeline = None
        self.trained = False
        self.classes_ = None

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn preprocessing + classifier pipeline."""

        # Encode categorical features as ordinal integers
        # handle_unknown="use_encoded_value" + unknown_value=-1
        # means unseen commands at inference time get -1 (safe fallback)
        categorical_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        preprocessor = ColumnTransformer(transformers=[
            ("cat", categorical_transformer, self.CATEGORICAL),
            ("num", "passthrough",           self.NUMERIC),
        ])

        if self.model_type == "logreg":
            classifier = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                C=1.0,
            )
        else:
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )

        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier",   classifier),
        ])

    def prepare(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and labels from the training pairs DataFrame.

        Fills missing prev_2/prev_3 values with the string "<START>"
        so the model treats them as a known token rather than NaN.

        Returns:
            X: Feature DataFrame
            y: Encoded integer labels
        """
        df = df.copy()

        for col in self.CATEGORICAL:
            if col in df.columns:
                df[col] = df[col].fillna("<START>")
            else:
                df[col] = "<START>"

        X = df[self.FEATURE_COLS]
        y_raw = df["target"]

        # Encode string command labels to integers
        y = self.label_encoder.fit_transform(y_raw)
        self.classes_ = self.label_encoder.classes_

        return X, y

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train the model and return evaluation metrics.

        Args:
            df:        Training pairs DataFrame from pipeline.
            test_size: Fraction of data to hold out for evaluation.

        Returns:
            dict with accuracy and classification report.
        """
        X, y = self.prepare(df)

        # With small datasets skip the split — train on everything
        if len(df) < 50:
            print(f"Small dataset ({len(df)} rows) — training on full data, skipping test split.")
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X, y)
            self.trained = True
            train_acc = accuracy_score(y, self.pipeline.predict(X))
            print(f"Training accuracy: {train_acc:.2%}")
            return {"train_accuracy": train_acc}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.trained = True

        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Model: {self.model_type}")
        print(f"Test accuracy:  {acc:.2%}")
        print(f"Train samples:  {len(X_train)}")
        print(f"Test samples:   {len(X_test)}")

        return {"accuracy": acc}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, last_commands: list[str], hour: int, day_of_week: int,
                directory: str, top_n: int = 3) -> list[dict]:
        """
        Predict the next command given context features.

        Args:
            last_commands: Recent commands, newest last. e.g. ["git add", "git status"]
            hour:          Current hour (0-23).
            day_of_week:   Current day (0=Monday).
            directory:     Normalized working directory.
            top_n:         Number of predictions to return.

        Returns:
            List of dicts: [{"command": "git commit", "probability": 0.85}, ...]
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Build feature row — pad with <START> if not enough history
        prev = list(reversed(last_commands[-3:]))
        while len(prev) < 3:
            prev.append("<START>")

        X = pd.DataFrame([{
            "prev_1":      prev[0],
            "prev_2":      prev[1],
            "prev_3":      prev[2],
            "hour":        hour,
            "day_of_week": day_of_week,
            "directory":   directory,
        }])

        proba = self.pipeline.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:top_n]

        return [
            {
                "command":     self.classes_[i],
                "probability": round(float(proba[i]), 3),
            }
            for i in top_indices
            if proba[i] > 0.0
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = None) -> None:
        if path is None:
            path = MODEL_PATH / f"sklearn_{self.model_type}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, model_type: str = "logreg", path: Path = None) -> "SklearnPredictor":
        if path is None:
            path = MODEL_PATH / f"sklearn_{model_type}.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded ← {path}")
        return model
import os
import pandas as pd
from typing import Dict, Any, Tuple
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

class SentimentAnalyzer:
    """Performs real-time sentiment detection and quantitative model evaluation."""
    
    def __init__(self):
        # Ensure VADER lexicon is available
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyzes a single user text and returns categorical sentiment, scores, and tone strategy."""
        if not text or not text.strip():
            return {
                "label": "neutral",
                "compound": 0.0,
                "scores": {"pos": 0.0, "neu": 1.0, "neg": 0.0},
                "tone_guidance": "Provide polite, factual, and direct customer assistance.",
                "badge_color": "gray"
            }

        scores = self.sia.polarity_scores(text)
        compound = scores["compound"]

        # Standard VADER compound thresholds
        if compound >= 0.05:
            label = "positive"
            tone_guidance = (
                "The customer is cheerful and satisfied. Respond with warm appreciation, "
                "enthusiasm, and courteous professionalism."
            )
            badge_color = "#28a745"
        elif compound <= -0.05:
            label = "negative"
            tone_guidance = (
                "EMPATHY & DE-ESCALATION REQUIRED: The customer is frustrated, upset, or unhappy. "
                "Acknowledge their feelings immediately, apologize sincerely for their trouble, "
                "and offer immediate, reassuring, and constructive solutions."
            )
            badge_color = "#dc3545"
        else:
            label = "neutral"
            tone_guidance = (
                "The customer has an objective inquiry. Provide clear, concise, "
                "and professional information."
            )
            badge_color = "#6c757d"

        return {
            "label": label,
            "compound": round(compound, 3),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "tone_guidance": tone_guidance,
            "badge_color": badge_color
        }

    def evaluate_dataset(self, csv_path: str = "datasets/sentiment_test.csv") -> Dict[str, Any]:
        """Evaluates the sentiment model on the test dataset and calculates full metrics."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Evaluation dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "sentiment" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'sentiment' columns.")

        y_true = df["sentiment"].astype(str).str.strip().str.lower().tolist()
        y_pred = []
        details = []

        for _, row in df.iterrows():
            text = str(row["text"])
            true_label = str(row["sentiment"]).strip().lower()
            res = self.analyze(text)
            pred_label = res["label"]
            y_pred.append(pred_label)
            details.append({
                "text": text,
                "true_sentiment": true_label,
                "predicted_sentiment": pred_label,
                "compound_score": res["compound"],
                "correct": true_label == pred_label
            })

        # Calculate standard classification metrics
        labels = ["positive", "neutral", "negative"]
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

        prec_weighted = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        return {
            "total_samples": len(df),
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "confusion_matrix": cm.tolist(),
            "labels": labels,
            "classification_report": report_dict,
            "details": details
        }

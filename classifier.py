import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin


label_names = {
    0: "Other",
    1: "Fault",
    2: "I/O",
    3: "Maintenance"
}

def domain_features(X):
    tags = X["plc_tag"]

    second_token = tags.str.split("_").str[1].fillna("")

    return pd.DataFrame({
        "has_flt": tags.str.contains(
            "FLT", case=False, na=False
        ).astype(int),

        "has_fault": tags.str.contains(
            "FAULT", case=False, na=False
        ).astype(int),

        "has_emg": tags.str.contains(
            "EMG", case=False, na=False
        ).astype(int),

        "has_fb": tags.str.contains(
            r"(?:^|_)FB(?:_|$)",
            case=False,
            regex=True,
            na=False
        ).astype(int),

        "has_op_suffix": tags.str.endswith(
            "_OP",
            na=False
        ).astype(int),

        "has_ip_suffix": tags.str.endswith(
            "_IP",
            na=False
        ).astype(int),

        "has_maint": tags.str.contains(
            "MAINT",
            case=False,
            na=False
        ).astype(int),

        "has_maintenance": tags.str.contains(
            "MAINTENANCE",
            case=False,
            na=False
        ).astype(int),

        "has_pc_maintenance": (
            tags.str.startswith("PC_", na=False) &
            second_token.str.upper().isin(
                ["M", "MAIN", "MAINT", "MAINTENANCE"]
            )
        ).astype(int),

        "has_alarm": tags.str.contains(
            r"(?:^|_)ALARM(?:_|$)",
            case=False,
            regex=True,
            na=False
        ).astype(int)
    }).values

class CharAndDomainFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1
        )

    def fit(self, X, y=None):
        self.tfidf.fit(X["plc_tag"])
        return self

    def transform(self, X):
        char_features = self.tfidf.transform(
            X["plc_tag"]
        )

        domain = domain_features(X)

        return hstack([
            char_features,
            domain
        ])

hybrid_model = Pipeline([
    ("features", CharAndDomainFeatures()),
    ("clf", LogisticRegression(
        max_iter=2000
    ))
])

combo_cols = [
    "has_flt",
    "has_fault",
    "has_emg",
    "has_fb",
    "has_op_suffix",
    "has_ip_suffix",
    "has_maint",
    "has_maintenance",
    "has_pc_maintenance",
    "has_alarm"
]


def classify_domain(tag_row):
    """
    Classifies a tag using validated domain signals.

    Returns:
        label  : 0, 1, 2, 3 or None
        status : "known", "conflict", or "unknown"
    """

    features = domain_features(
        pd.DataFrame([tag_row])
    )[0]

    feature_dict = dict(
        zip(combo_cols, features)
    )

    active = []

    if feature_dict["has_flt"]:
        active.append(("FLT", 1))

    if feature_dict["has_fault"]:
        active.append(("FAULT", 1))

    if feature_dict["has_emg"]:
        active.append(("EMG", 1))

    if feature_dict["has_fb"]:
        active.append(("FB", 2))

    if feature_dict["has_op_suffix"]:
        active.append(("_OP", 2))

    if feature_dict["has_ip_suffix"]:
        active.append(("_IP", 2))

    if (
        feature_dict["has_maint"]
        or feature_dict["has_maintenance"]
        or feature_dict["has_pc_maintenance"]
    ):
        active.append(("MAINTENANCE", 3))

    if feature_dict["has_alarm"]:
        active.append(("ALARM", 1))

    # No known signal
    if not active:
        return None, "unknown"

    # Conflicting signals
    labels = set(
        label for _, label in active
    )

    if len(labels) > 1:
        return None, "conflict"

    # All active signals agree
    label = active[0][1]

    return label, "known"

def predict_tags(input_df, model):
    results = []

    for idx, row in input_df.iterrows():

        domain_label, domain_status = classify_domain(row)

        # Clear domain signal
        if domain_status == "known":

            prediction = domain_label
            source = "Domain"
            review_required = False
            review_reason = ""

        # Conflicting domain signals
        elif domain_status == "conflict":

            prediction = None
            source = "Domain Conflict"
            review_required = True
            review_reason = "Conflicting domain signals"

        # No domain signal → ML
        else:

            prediction = int(
                model.predict(
                    pd.DataFrame([row])
                )[0]
            )

            source = "ML"
            review_required = False
            review_reason = ""

        results.append({
            "plc_tag": row["plc_tag"],
            "access": row["access"],
            "data_type": row["data_type"],
            "predicted_label": prediction,
            "category": (
                label_names[prediction]
                if prediction is not None
                else "Pending Review"
            ),
            "source": source,
            "review_required": review_required,
            "review_reason": review_reason
        })

    return pd.DataFrame(results)

from io import BytesIO


def create_output_excel(results_df):

    output = BytesIO()

    # Final user-facing columns
    final_df = results_df[
        [
            "plc_tag",
            "access",
            "data_type",
            "category"
        ]
    ].copy()

    # Rename columns
    final_df.columns = [
        "PLC Tag Name",
        "Access Type",
        "Data Type",
        "Tag Category"
    ]

    sheet_names = {
        "Fault": "Fault",
        "I/O": "IO",
        "Maintenance": "Maintenance",
        "Other": "Other"
    }

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # Complete classified list
        final_df.to_excel(
            writer,
            sheet_name="All_Tags",
            index=False
        )

        # Category-wise sheets
        for category, sheet_name in sheet_names.items():

            category_df = final_df[
                final_df["Tag Category"] == category
            ]

            category_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

        # Pending review sheet
        pending_df = final_df[
            results_df["review_required"] == True
        ]

        pending_df.to_excel(
            writer,
            sheet_name="Pending_Review",
            index=False
        )

    output.seek(0)

    return output

MODEL_PATH = "hybrid_model.joblib"

model = joblib.load(MODEL_PATH)

print("Hybrid model loaded successfully.")
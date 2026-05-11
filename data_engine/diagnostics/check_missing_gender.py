"""
Temporary script: find exams used in NIfTI training data that are missing
a valid gender value (0 or 1) in classes.csv.

Usage:
    python data_engine/check_missing_gender.py
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

classes    = pd.read_csv(BASE / "dataset/classes.csv")
survey     = pd.read_csv(BASE / "diagnostics/outputs/nifti_survey.csv")

used = classes[classes["exam"].isin(survey["exam"])].copy()

invalid = used[~used["gender"].isin([0, 1])]

print(f"Exams in nifti_survey : {len(survey)}")
print(f"Matched in classes.csv: {len(used)}")
print(f"Missing/invalid gender: {len(invalid)}\n")
print(invalid[["exam", "class", "Age", "gender"]].to_string(index=False))

"""
Validation check: split_data must be deterministic given
the same seed — same train/val/test membership across runs.

Run: python training_engine/tests/test_split_consistency.py
"""

from src.data import split_data


def demo() -> None:
    data = [
        {"image": f"/fake/{i}.nii.gz", "label": i % 2, "path": f"/fake/{i}.nii.gz"}
        for i in range(50)
    ]

    def run():
        return [
            (fold, [d["path"] for d in tr], [d["path"] for d in va], [d["path"] for d in te])
            for fold, tr, va, te in split_data(data, n_splits=5, seed=42)
        ]

    a, b = run(), run()
    assert a == b, "split_data produced different splits across runs with the same seed"
    print("OK: split_data is deterministic across repeated calls with the same seed.")


if __name__ == "__main__":
    demo()

from pathlib import Path

import pandas as pd


def find_project_root():
    current = Path(__file__).resolve().parent

    for directory in (current, *current.parents):
        if (
            (directory / "configs").is_dir()
            and (directory / "src").is_dir()
            and (directory / "requirements.txt").is_file()
        ):
            return directory

    raise RuntimeError(
        "Could not locate project root. "
        "Expected configs/, src/, and requirements.txt."
    )


PROJECT_ROOT = find_project_root()

OUTPUT = PROJECT_ROOT / "results" / "leave_one_patient_out"
INPUT = OUTPUT / "leave_one_patient_out_all.csv"
SUMMARY_INPUT = OUTPUT / "leave_one_patient_out_summary.csv"

METHODS = [
    "equal_weight_per_seed",
    "equal_weight_per_patient",
    "equal_weight_per_unique_image",
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.is_file():
        raise FileNotFoundError(
            f"LOPO results file not found: {INPUT}. "
            "Run leave_one_patient_out.py first."
        )

    if not SUMMARY_INPUT.is_file():
        raise FileNotFoundError(
            f"LOPO summary file not found: {SUMMARY_INPUT}. "
            "Run leave_one_patient_out.py first."
        )

    df = pd.read_csv(INPUT)
    summary_df = pd.read_csv(SUMMARY_INPUT)

    required_result_columns = {"metric", "excluded_patient", *METHODS}
    missing_result = required_result_columns - set(df.columns)
    if missing_result:
        raise ValueError(
            f"Missing columns in LOPO results: {sorted(missing_result)}"
        )

    required_summary_columns = {
        "metric",
        *(f"{method}_baseline" for method in METHODS),
    }
    missing_summary = required_summary_columns - set(summary_df.columns)
    if missing_summary:
        raise ValueError(
            f"Missing columns in LOPO summary: {sorted(missing_summary)}"
        )

    # Read baseline values from the LOPO summary instead of maintaining
    # manually copied constants that could become stale.
    baseline = {}
    for _, row in summary_df.iterrows():
        metric = row["metric"]
        baseline[metric] = {
            method: row[f"{method}_baseline"]
            for method in METHODS
        }

    ranked = []

    for metric in baseline:
        subset = df[df["metric"] == metric].copy()

        if subset.empty:
            print(f"Skipping {metric}: no LOPO rows found.")
            continue

        for method in METHODS:
            base = baseline[metric][method]
            subset["absolute_change"] = (subset[method] - base).abs()
            subset["direction_changed"] = (
                (subset[method] > 0) != (base > 0)
            )

            top = subset.nlargest(5, "absolute_change")

            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                ranked.append({
                    "metric": metric,
                    "method": method,
                    "rank": rank,
                    "excluded_patient": row["excluded_patient"],
                    "baseline": base,
                    "value_after_exclusion": row[method],
                    "absolute_change": row["absolute_change"],
                    "direction_changed": row["direction_changed"],
                })

    result = pd.DataFrame(ranked)
    output_file = OUTPUT / "influential_patients.csv"
    result.to_csv(output_file, index=False)

    for metric in baseline:
        print(f"\n{'=' * 65}")
        print(metric)

        for method in METHODS:
            print(f"\n{method}")
            part = result[
                (result["metric"] == metric)
                & (result["method"] == method)
            ]
            print(
                part[
                    [
                        "rank",
                        "excluded_patient",
                        "value_after_exclusion",
                        "absolute_change",
                        "direction_changed",
                    ]
                ].to_string(index=False)
            )

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()

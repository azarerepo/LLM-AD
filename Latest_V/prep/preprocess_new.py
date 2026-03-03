import sys
import pandas as pd
from collections import Counter
import pprint
import numpy as np
import json


def to_nan(val):
    # treat '', None, or already NaN as np.nan
    if pd.isnull(val) or val == "" or val is None:
        return np.nan
    return val


def safe_value(val, for_prompt=False):
    # Returns np.nan for DataFrame, or "missing"/"N/A" for prompt
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "missing" if for_prompt else np.nan
    return val


def flatten_alzheimers_json(
    json_file, out_csv, static_in_prompt=True, add_missing_mask=False
):
    with open(json_file, "r") as f:
        data = json.load(f)

    feature_names = set()
    static_feature_names = set()
    # Collect feature and static names for columns
    for subj in data:
        static_feature_names |= set(subj["demographics"].keys())
        for feats in subj["time_series"].values():
            feature_names |= set(feats.keys())
    feature_names = sorted(feature_names)
    static_feature_names = sorted(static_feature_names)

    rows = []
    for subj in data:
        statics = {**subj["demographics"], "subject_id": subj["subject_id"]}
        for visit, feats in subj["time_series"].items():
            row = {"subject_id": subj["subject_id"], "visit_month": visit}
            if not static_in_prompt:
                row.update(statics)
            for fname in feature_names:
                row[fname] = safe_value(feats.get(fname, np.nan))
                if add_missing_mask:
                    row[f"{fname}_is_missing"] = int(feats.get(fname, None) is None)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


def main(csv_file, output_json):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Specify required visits and time series columns
    # timepoints = ["bl", "m06", "m12", "m18", "m24", "m36", "m48"]
    timepoints = ["visit1", "visit2", "visit3",
                  "visit4", "visit5", "visit6", "visit7"]
    # ts_vars = ["CDRSB", "CDGLOBAL", "TOTAL13", "FAQTOTAL", "MMSCORE", "AVDEL30MIN"]
    # ts_vars = ['CDRSUM', 'CDRGLOB', 'NACCMMSE', 'FAQTOTAL', 'REYDREC']
    ts_vars = ["CDRSB", "CDGLOBAL", "FAQTOTAL", "MMSCORE", "MOCA", "REYDREC"]

    # Pre-calculate RAVLT_immediate for all rows
    # df["RAVLT_immediate"] = df[
    #     ["AVTOT1", "AVTOT2", "AVTOT3", "AVTOT4", "AVTOT5"]
    # ].apply(lambda row: pd.to_numeric(row, errors="coerce").sum(), axis=1)


    # Prepare demographic data from "sc" visit (screening)
    # demog_cols = ["subject_id", "subject_age", "PTGENDER", "GENOTYPE", "PTEDUCAT"]
    demog_cols = ["subject_id", "subject_age", "PTGENDER", "APOE", "APOE4CNT", "PTEDUCAT"]
    # demog_cols = ['subject_id', 'NACCAGE', 'NACCAGEB', 'SEX', 'EDUC']

    # demog_df = df[df["visit"] == "sc"].copy()

    demog_df = df[df["visit"] == "visit1"].copy()

    demog_df = demog_df[demog_cols].drop_duplicates("subject_id")
    demog_df.set_index("subject_id", inplace = True)

    # Initialize final structure
    all_subjects = []

    # Group by subject
    for subject, sub_df in df.groupby("subject_id"):
        # Get demographic info
        if subject in demog_df.index:
            demo = demog_df.loc[subject].to_dict()
        else:
            continue  # skip subject if no screening row

        # For each required timepoint, extract time series values
        ts_records = {}
        for tp in timepoints:
            row = sub_df[sub_df["visit"] == tp]
            if not row.empty:
                rec = row.iloc[0]
                ts_records[tp] = {var: to_nan(rec.get(var, None)) for var in ts_vars}
                # ts_records[tp]["RAVLT_immediate"] = to_nan(
                #     rec.get("RAVLT_immediate", None)
                # )
            else:
                # ts_records[tp] = {var: np.nan for var in ts_vars + ["RAVLT_immediate"]}
                ts_records[tp] = {var: np.nan for var in ts_vars}
        missing_cdrsb = sum(
            pd.isnull(ts_records[tp]["CDRSB"])
            or str(ts_records[tp]["CDRSB"]).strip() == ""
            for tp in timepoints
        )

        # Only keep subjects with fewer than 6 missing CDRSB
        if missing_cdrsb < 6:
            all_subjects.append(
                {
                    "subject_id": subject,
                    "demographics": demo,
                    "time_series": ts_records,
                    "missing_cdrsb_count": missing_cdrsb,
                }
            )

    # Count subjects by number of missing CDRSB
    missing_counts = Counter(sub["missing_cdrsb_count"] for sub in all_subjects)
    print("Number of subjects by missing CDRSB count across 7 timepoints:")
    for miss_count in sorted(missing_counts):
        print(f"{miss_count} missing: {missing_counts[miss_count]} subjects")

    print("\nSubject IDs for each missing CDRSB count category:")
    by_missing = {}
    for sub in all_subjects:
        count = sub["missing_cdrsb_count"]
        by_missing.setdefault(count, []).append(sub["subject_id"])
    for count in sorted(by_missing):
        subjects = by_missing[count]
        print(f"{count} missing ({len(subjects)} subjects):")
        print(", ".join(subjects[:2]))

    print(
        "\nFull sample data for each missing CDRSB count category (first two per category):"
    )
    by_missing_full = {}
    for sub in all_subjects:
        count = sub["missing_cdrsb_count"]
        by_missing_full.setdefault(count, []).append(sub)

    pp = pprint.PrettyPrinter(indent=2)
    for count in sorted(by_missing_full):
        samples = by_missing_full[count][:2]  # first two samples
        print(f"\n--- {count} missing ({len(by_missing_full[count])} subjects):")
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            pp.pprint(sample)

    # Save output JSON
    with open(output_json, "w") as f:
        # Convert all np.nan to None for JSON
        def clean(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean(x) for x in obj]
            return obj

        json.dump([clean(s) for s in all_subjects], f, indent=2)
    print(f"Saved {len(all_subjects)} subjects to {output_json}")

    # Reload for double-check
    with open(output_json, "r") as f:
        reloaded = json.load(f)
    print(f"Reloaded {len(reloaded)} subjects from {output_json}.")
    print("First reloaded subject for sanity check:")

    pp = pprint.PrettyPrinter(indent=2, width=120)
    pp.pprint(reloaded[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process and flatten Alzheimer’s dataset."
    )
    parser.add_argument("input_csv", help="Path to raw CSV (e.g. adni.csv)")
    parser.add_argument("parsed_json", help="Path to output parsed JSON")
    parser.add_argument(
        "--flatten_csv",
        default=None,
        help="If set, flatten JSON to CSV for TimeLLM here.",
    )
    parser.add_argument(
        "--static_in_prompt",
        action="store_true",
        help="Put statics in prompt instead of features",
    )
    parser.add_argument(
        "--add_missing_mask",
        action="store_true",
        help="Add is_missing mask columns in CSV",
    )

    args = parser.parse_args()

    # Step 1: CSV → parsed JSON
    main(args.input_csv, args.parsed_json)

    # Step 2: parsed JSON → flattened CSV (optional)
    if args.flatten_csv:
        flatten_alzheimers_json(
            args.parsed_json,
            args.flatten_csv,
            static_in_prompt=args.static_in_prompt,
            add_missing_mask=args.add_missing_mask,
        )


# Usage example:
# python alz_pipeline.py raw_adni.csv parsed_adni.json --flatten_csv alzheimers_timeseries.csv --static_in_prompt --add_missing_mask
# python ./src/preprocess.py ./data/raw/CDRSB_Cohort_My_Table_14Jun2025.csv ./data/processed/parsed_adni.json --flatten_csv ./data/processed/alzheimers_timeseries.csv --static_in_prompt --add_missing_mask

# python ./src/preprocess.py ./data/raw/Longitudinal_My_Table_13Jun2025.csv
# python ./src/preprocess.py /path/to/adni.csv /path/to/parsed_adni.json
# Check for a subject:
# print(subjects["002_S_0619"]["predictors"])  # PTEDUCAT guaranteed consistent
# print(subjects["002_S_0619"]["time_series"])  # starts at bl=0, then m06, m12…
# python ./src/preprocess.py ./data/raw/CDRSB_Cohort_My_Table_14Jun2025.csv ./data/processed/parsed_CDRSB_Cohort_My_Table_14Jun2025.json

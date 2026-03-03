import pandas as pd

# Load the results CSV file
file_name = "results_Exp_12lyrs_trn0.7_val0.0_LR0.005_bs16_FMMSCORE,FAQTOTAL,CDRSB_SFsubject_age_itr1.csv"
file_path = "./New_Results/"  # Adjust path as needed
df = pd.read_csv(file_path + file_name)


# List of clinical target MAE columns
prefix = "test_mae_"
clin_targ = [s for s in list(df.columns) if s.startswith(prefix)]
clin_targ[0], clin_targ[1] = clin_targ[1], clin_targ[0]
# clin_targ = [
#     "test_mae_FAQTOTAL",
#     "test_mae_MMSCORE",
#     "test_mae_CDRSB",
# ]

# Group by visit order and LLM model
group_cols = ["visit_order", "llm_model"]
aggregated_results = (
    df.groupby(group_cols)[clin_targ].agg(["mean", "std"]).round(4)
)

# Flatten MultiIndex columns
aggregated_results.columns = [
    "_".join(col).strip() for col in aggregated_results.columns.values
]
aggregated_results.reset_index(inplace = True)

# Save to CSV
aggregated_results.to_csv(file_path +
    "Aggregated_mae_by_visit_" + file_name,
    index=False,
)

print("Saved summary to 'aggregated_mae_by_visit_llm.csv'")
print(aggregated_results.head())

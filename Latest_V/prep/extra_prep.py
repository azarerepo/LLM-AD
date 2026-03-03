import pandas as pd
import numpy as np


### replaces age at baseline with age_at_visit
### Adds months_from_baseline, and reorders columns

### Correct order:
# 1) read_data_nacc.py
# 2) preprocess_new.py
# 3) This script: extra_prep.py


path_to_src = './data/raw/'
src_file = 'investigator_ftldlbd_nacc71_reduced12.csv'
# src_file = 'CDRSB_Cohort_My_Table_14Jun2025.csv'

path_to_saved = './data/processed/'
saved_file = 'nacc_timeseries.csv'
# saved_file = 'alzheimers_timeseries.csv'

# Read the CSV files into a DataFrame
df_src = pd.read_csv(path_to_src + src_file, low_memory = False)
df_saved = pd.read_csv(path_to_saved + saved_file, low_memory = False)

if len(df_src) != len(df_saved) or (df_saved['subject_id'] != df_src['subject_id']).any():
  raise Exception("The two dataframes must have the same number of rows!")


df_saved["subject_age"] = df_src["subject_age"].copy()
df_saved["months_from_baseline"] = df_src["months_from_baseline"].copy()


# map APOE values to x/y format
apoe4_map = {1.0: "3/3", 2.0: "3/4", 3.0: "3/2",
             4.0: "4/4", 5.0: "4/2", 6.0: "2/2"}
df_saved["APOE_numeric"] = df_saved["APOE"].copy()
df_saved["APOE"] = df_saved["APOE"].map(apoe4_map)


# Re-order columns
new_cols = ["subject_id","visit_month", "months_from_baseline",
            "subject_age","PTGENDER",
            "APOE","APOE_numeric","APOE4CNT","PTEDUCAT",
            "CDGLOBAL","CDGLOBAL_is_missing",
            "CDRSB","CDRSB_is_missing",
            "FAQTOTAL","FAQTOTAL_is_missing",
            "MMSCORE","MMSCORE_is_missing",
            "MOCA", "MOCA_is_missing",
            "REYDREC","REYDREC_is_missing"]
df_saved = df_saved.loc[:, new_cols]

df_saved.to_csv(path_to_saved + 'nacc_timeseries_final.csv', index = False)

print(f'\nEnd of code!\n')
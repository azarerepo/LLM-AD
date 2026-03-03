import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


df = pd.read_csv('./data/raw/investigator_ftldlbd_nacc71.csv', low_memory = False)
# df = pd.read_csv('./data/raw/CDRSB_Cohort_My_Table_14Jun2025.csv')

############################################################
# number of visits per participant

visit_counts = np.array([])
for subj, sub_df in tqdm(df.groupby('NACCID'), desc = 'Subjects'):
    visit_counts = np.append(visit_counts, int(sub_df['NACCAVST'].values[0]))

print(f'\nVisits per participant for {len(visit_counts)} subjects:')
print('Mean no. visits: ', visit_counts.mean())
print('STD: ', visit_counts.std())
print("")

############################################################
### keep only subjects with a certain of number of visits

n_visit = 7
# df_sel = df.loc[(df['NACCAVST'] >= 3) & (df['NACCAVST'] <= 7),:].copy()
# df_sel = df[(df['NACCAVST'] >= 3) & (df['NACCAVST'] <= 7)].copy()
# df_sel = df[df['NACCAVST'].between(3, 7)].copy()
df_sel = df.loc[df['NACCAVST'] == n_visit,:].copy()

############################################################
# Some cleaning...

col_names = df_sel.columns

# partial or total matches for timeseries variables of interest
ts_vars = ["CDR", "CDRSB", "CDGLOBAL", 
           "AV", "TOT", "REYDREC",
           "TOTAL", "GLOBAL",
           "FAQ", "MM", "SCORE", "AVDEL", "MOCA"]
matching_items = []
for ts_var in ts_vars:
    matching_items.append(
        [item for item in col_names if (ts_var in item or item in ts_var)]
        )
mtch_itm_flt = [item for sublist in matching_items for item in sublist]


FAQ_vars = ['BILLS', 'TAXES', 'SHOPPING',
            'GAMES', 'STOVE', 'MEALPREP',
            'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']

demog_vars = ['NACCID', 'NACCAPOE', 'NACCNE4S',
              'VISITMO', 'VISITDAY', 'VISITYR',
              'NACCAGE', 'NACCAGEB', 'SEX', 'EDUC']

cols_init = demog_vars + mtch_itm_flt + FAQ_vars

df_sel = df_sel[cols_init]


# Create a new column containing the sum of all FAQs
df_sel['FAQTOTAL'] = df_sel[FAQ_vars].apply(
    lambda row: pd.to_numeric(row, errors="coerce").sum(), axis = 1
    )
df_sel.drop(FAQ_vars, axis = 1)

cols_reduced = ['CDRSUM', 'CDRGLOB',
                'NACCMMSE', 'NACCMOCA',
                'FAQTOTAL', 'REYDREC']
df_sel = df_sel[demog_vars + cols_reduced]

#######################################################
### Sort visits from oldest to most recent for each subject
# and create labels for visits

df_sel["visit_date"] = pd.to_datetime(
    dict(
        year = df_sel["VISITYR"],
        month = df_sel["VISITMO"],
        day = df_sel["VISITDAY"]
    )
)
df_sel = df_sel.sort_values(["NACCID", "visit_date"])
df_sel["visit"] = (
    df_sel.groupby("NACCID")
    .cumcount()
    .add(1)
    .astype(str)
    .radd("visit")
)


### Get information about the spread of visits
# Get first visit date per subject
first_visit = df_sel.groupby("NACCID")["visit_date"].transform("min")

# # Compute month difference since first visit
# df_sel["month_count"] = (
#     (df_sel["visit_date"].dt.year - first_visit.dt.year) * 12 +
#     (df_sel["visit_date"].dt.month - first_visit.dt.month)
# )

# Calendar month difference
df_sel["months_from_baseline"] = (
    df_sel["visit_date"].dt.to_period("M").astype(int) -
    first_visit.dt.to_period("M").astype(int)
)
# Force strict monotonic increase
df_sel["months_from_baseline"] = (
    df_sel.groupby("NACCID")["months_from_baseline"]
    .cummax()
)

diffs = df_sel["months_from_baseline"].diff().values
diffs[np.isnan(diffs)] = 0
diffs[diffs < 0] = 0
diffs_nz = diffs[diffs > 0].copy()

fig, ax = plt.subplots(figsize = (8, 6))
freqs, _, _ = ax.hist(diffs_nz, bins = 50, color = 'skyblue', edgecolor='black')
x_value = diffs_nz.mean()
plt.axvline(x = diffs_nz.mean(),
            color = 'red',
            linestyle = '--',
            linewidth = 1.5)
plt.annotate(
    f'Mean = {diffs_nz.mean():.1f}',
    xy = (diffs_nz.mean(), 100), # point to annotate
    xytext = (diffs_nz.mean() + 5, freqs.max()*0.95), # text position
    # arrowprops = dict(facecolor = 'red', shrink = 0.05),
    color = 'red',
    fontsize = 16
)
plt.axvline(x = np.median(diffs_nz),
            color = 'blue',
            linestyle = '--',
            linewidth = 1.5)
plt.annotate(
    f'Median = {np.median(diffs_nz):.1f}',
    xy = (np.median(diffs_nz), 100), # point to annotate
    xytext = (np.median(diffs_nz) + 5, freqs.max()*0.9), # text position
    # arrowprops = dict(facecolor = 'red', shrink = 0.05),
    color = 'blue',
    fontsize = 16
)
ax.tick_params(axis = 'both', labelsize = 16)
ax.set_title('Consecutive Visit Differences', fontsize = 16)
ax.set_xlabel('Visit Differences (months)', fontsize = 16)
ax.set_ylabel('Frequency', fontsize = 16)
ax.grid(axis='y', alpha = 0.75)
plt.show(block = False)


### Find subjects whose consecutive visits are not apart by
# more than a certain amount
m1, m2 = 9, 15 # minimum and maximum amount in months consecutive visits can be apart
mask = ((diffs >= m1) & (diffs <= m2)) | (diffs == 0)
reg_vis_num = np.array([])
reg_vis_IDs = []
for subj, sub_df in (df_sel[mask]).groupby('NACCID'):
    reg_vis_num = np.append(reg_vis_num, len(sub_df))
    reg_vis_IDs.append(subj)

fig, ax = plt.subplots(figsize = (8, 6))
bins = np.arange(reg_vis_num.min() - 0.5, reg_vis_num.max() + 1.5, 1)
freqs, _, _ = ax.hist(reg_vis_num, bins = bins, color = 'skyblue', edgecolor='black')
plt.axvline(x = reg_vis_num.mean(),
            color = 'red',
            linestyle = '--',
            linewidth = 2)
plt.annotate(f'Median = {np.median(reg_vis_num):.1f}',
             xy = (np.median(reg_vis_num), 100), # point to annotate
             xytext = (0.8*x_value, freqs.max()*0.95), # text position
             # arrowprops = dict(facecolor = 'red', shrink = 0.05),
             color = 'red')
ax.set_title(f'Visit count for subjects with consecutive visits apart by {m1} to {m2} months.')
ax.set_xlabel('Number of Visits')
ax.set_ylabel('Frequency')
ax.grid(axis='y', alpha = 0.75)
plt.show(block = False)

###################################################################
### Remove missing data flagged with -4, 9, <0, or other
# numbers used to identify inadmissible data
print('Removing inadmissible values...')

for col in df_sel.columns:
    df_sel.loc[df_sel[col] == -4, col] = None

df_sel.loc[df_sel['NACCAPOE'] == 9, 'NACCAPOE'] = None
df_sel.loc[df_sel['NACCNE4S'] == 9, 'NACCNE4S'] = None

df_sel.loc[df_sel['FAQTOTAL'] < 0, 'FAQTOTAL'] = None

df_sel.loc[df_sel['NACCMOCA'] == 99, 'NACCMOCA'] = None
df_sel.loc[df_sel['NACCMOCA'] == 88, 'NACCMOCA'] = None

df_sel.loc[df_sel['NACCMMSE'] == 88, 'NACCMMSE'] = None
df_sel.loc[df_sel['NACCMMSE'].between(95, 98), 'NACCMMSE'] = None

print('Done!')

###################################################################
### Use MOCA to fill in missing values in MMSE

map_to_mmse = {2:7, 3:9, 4:10, 5:12, 6:13, 7:14, 8:15, 9:16,
               10:17, 11:18, 12:19, 13:20, 14:21, 15:22, 16:23,
               17:24, 18:25, 19:25, 20:26, 21:27, 22:27, 23:28,
               24:28, 25:29, 26:29, 27:30, 28:30, 29:30, 30:30}

mask = df_sel['NACCMMSE'].isna() & df_sel['NACCMOCA'].isin(map_to_mmse)
df_sel.loc[mask, 'NACCMMSE'] = df_sel.loc[mask, 'NACCMOCA'].map(map_to_mmse)

###################################################################
### Create all subsets of selected features

def powerset(iterable):
    """
    Ex. powerset([1,2,3]) results in () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    from itertools import chain, combinations
    s = list(iterable)
    # Generate combinations of all sizes (from 0 to the length of the set)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def counts_info(df, set_list, col_name = 'NACCID'):
    """
    Inputs:
    df: dataframe
    set_list: list of lists: a list of subsets of the columns of df
    col_name: identifier for unique rows in df
    All unique rows have the same number of occurrences in df

    Output:
    A tensor of size (K, R, S). 
    K: no. unique subjects in df
    R: no. times each unique subject identifier is repeated
    S: no. lists in set_list

    Each output entry is 1 or 0: 1 if the data value exists, 0 otherwise
    """
    subj_count = len(np.unique(df[col_name]))
    counts = np.zeros((subj_count, n_visit, len(set_list)))
    age_bl = np.zeros((subj_count)) # age at baseline
    for k, sub_df in enumerate(df.groupby(col_name)):
        age_bl[k] = sub_df[1].iloc[0]['NACCAGE']
        for row in range(len(sub_df[1])):
            for i in range(len(set_list)):
                if sub_df[1].iloc[row][set_list[i]].notna().any():
                    counts[k,row,i] = counts[k,row,i] + 1
    return counts, age_bl

# pset_list = [list(subset) for subset in powerset(cols_reduced)]
# pset_list = pset_list[1:] # remove the empty set
pset_list = [cols_reduced] # only one chosen subset

counts, age_bl = counts_info(df_sel, pset_list)
counts_per_collist = (counts.sum(axis = 1) > 0).sum(axis = 0)
num_subj = counts_per_collist[pset_list.index(cols_reduced)]
#######################################################

### Rename some columns to match those of ADNI (for simplicity)

df_sel.rename(columns = {'NACCID': 'subject_id',
                       'CDRSUM': 'CDRSB',
                       'CDRGLOB': 'CDGLOBAL',
                       'NACCMMSE': 'MMSCORE',
                       'NACCMOCA': 'MOCA',
                       'NACCAGE': 'subject_age',
                       'SEX': 'PTGENDER',
                       'EDUC': 'PTEDUCAT',
                       'NACCAPOE': 'APOE',
                       'NACCNE4S': 'APOE4CNT'},
                       inplace = True)

#######################################################
### Turn int64 data into Python int if needed

# is_int64_series = (df_sel.dtypes == np.int64)
# int64_columns = is_int64_series[is_int64_series].index.tolist()
# for col in int64_columns:
#         if df_sel[col].dtype == 'int64':
#             # Cast the column to the standard Python int type
#             df_sel[col] = df_sel[col].astype(object).apply(lambda x: int(x) if pd.notna(x) else x)

int64_columns = df_sel.select_dtypes(include = ['int64']).columns
for col in int64_columns:
    df_sel[col] = (
        df_sel[col]
        .astype(object)
        .apply(lambda x: int(x) if pd.notna(x) else None)
    )
#######################################################


df_sel.to_csv('./investigator_ftldlbd_nacc71_reduced.csv', index = False)

print(f'\nEnd of code!\n')
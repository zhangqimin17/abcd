import pandas as pd

def generate_real_data(cutoff=65):
    df = pd.read_csv('abcd_betnet02.txt', sep="\t", skiprows=[1])
    outcome = pd.read_csv('abcd_cbcls01.txt', sep="\t", skiprows=[1])[['subjectkey', 'cbcl_scr_07_ocd_t']]
    outcome['outcome'] = (outcome['cbcl_scr_07_ocd_t'] >= cutoff).astype(int)
    df = pd.merge(df, outcome, how='left', on='subjectkey')
    
    predictive_cols = [col for col in df.columns if col.startswith('rsfmri') and 'visitid' not in col]
    # Fill missing values in each column with the mean of that column
    for col in predictive_cols:
        if df[col].dtype != 'object':  # Exclude string/object columns
            mean_col = df[col].mean()
            df[col] = df[col].fillna(mean_col)

    return df
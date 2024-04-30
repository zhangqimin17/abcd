import pandas as pd
import numpy as np

def generate_simulation_data(outcome_type_binary = True, num_rows=1000, num_numeric_columns=20, n_columns_to_make_for_each_column=4):
    np.random.seed(42)
    
    n_key_columns = int(num_numeric_columns/4)
    n_columns_to_make_correlated = int(num_numeric_columns/2)
    
 
    columns = ['X'+str(i) for i in range(1, num_numeric_columns+1)]
    data = np.random.randn(num_rows, num_numeric_columns)
    df = pd.DataFrame(data, columns=columns)
    
    if outcome_type_binary:
        # Generate coefficients for predictive columns
        coefficients1 = np.random.randn(n_key_columns)
        coefficients2 = np.random.randn(n_key_columns)
        # Create binary target column Y based on predictive columns
        df['Y'] = 0
        for i in range(n_key_columns):
            df['Y'] += coefficients1[i] * df['X'+str(i+1)]**2 + coefficients2[i] * df['X'+str(i+1)] + np.random.normal(0, 1, num_rows)

        # Apply threshold to convert target column to binary
        threshold = df['Y'].quantile(0.9)
        df['Y'] = (df['Y'] > threshold).astype(int)
    else:
        predictive_columns = np.random.choice(df.columns, n_key_columns, replace=False)
        coefficients = np.random.rand(n_key_columns)
        df['Y'] = np.sum(df[predictive_columns] * coefficients, axis=1) + np.random.normal(0, 1, num_rows)
        

    # Randomly select 10 features (excluding Y column)
    selected_features = np.random.choice(df.columns[:-1], size=n_columns_to_make_correlated, replace=False)

    # For each selected feature, create new columns to correlate with it
    for feature in selected_features:
        for i in range(1, n_columns_to_make_for_each_column+1):
            # Create new column with correlation to the selected feature plus some noise
            df[f'{feature}_correlated_{i}'] = df[feature] * (1 + np.random.normal(0, 1, num_rows)) + np.random.normal(0, 1, num_rows)

    return df
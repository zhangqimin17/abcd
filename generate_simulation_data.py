import pandas as pd
import numpy as np

def generate_simulation_data(num_rows=1000, num_numeric_columns=20, n_columns_to_make_for_each_column=4):
    np.random.seed(42)
    
    n_key_columns = int(num_numeric_columns/4)
    n_columns_to_make_correlated = int(num_numeric_columns/2)
    
 
    columns = ['X'+str(i) for i in range(1, num_numeric_columns+1)]
    data = np.random.randn(num_rows, num_numeric_columns)
    df = pd.DataFrame(data, columns=columns)

    # Generate coefficients for predictive columns
    coefficients = np.random.randn(n_key_columns)

    # Create binary target column Y based on predictive columns
    df['Y'] = 0
    for i in range(n_key_columns):
        df['Y'] += coefficients[i] * df['X'+str(i+1)] + np.random.normal(0, 1, num_rows)

    # Apply threshold to convert target column to binary
    threshold = df['Y'].quantile(0.5)
    df['Y'] = (df['Y'] > threshold).astype(int)

    # Randomly select 10 features (excluding Y column)
    selected_features = np.random.choice(df.columns[:-1], size=n_columns_to_make_correlated, replace=False)

    # For each selected feature, create new columns to correlate with it
    for feature in selected_features:
        for i in range(1, n_columns_to_make_for_each_column+1):
            # Create new column with correlation to the selected feature plus some noise
            df[f'{feature}_correlated_{i}'] = df[feature] * (1 + np.random.normal(0, 1, num_rows)) + np.random.normal(0, 1, num_rows)

    return df
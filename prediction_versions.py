import pandas as pd
import os

def update_predictions_csv(new_predictions_df, csv_path='predictions-05-19.csv', version_name=None):
    """
    Adds new predictions as a new version column in a single CSV file.
    new_predictions_df: DataFrame with columns ['Date', 'Stock', 'Prediction']
    version_name: Custom string for this prediction column, e.g. 'Prediction_10yTind'
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Use custom version name or auto-increment if not provided
        if version_name is None:
            pred_cols = [col for col in df.columns if col.startswith('Prediction_')]
            if pred_cols:
                last_version = max([int(col.split('_v')[-1]) for col in pred_cols if '_v' in col])
            else:
                last_version = 0
            version_name = f'Prediction_v{last_version+1}'
        # Merge on Date and Stock
        df = df.merge(new_predictions_df, on=['Date', 'Stock'], how='outer')
        df.rename(columns={'Prediction': version_name}, inplace=True)
    else:
        # First run: create the CSV
        if version_name is None:
            version_name = 'Prediction_v1'
        df = new_predictions_df.rename(columns={'Prediction': version_name})
    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved with new column: {version_name}")

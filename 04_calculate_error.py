import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true != 0) else 0

def calculate_metrics(file_path):
    """
    Reads a CSV file, calculates MAE, RMSE, MAPE, and R^2 score.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        metric_names = ["MAE", "RMSE", "MAPE", "R2"]

        if 'Metric' in df.columns:
            df = df[~df['Metric'].isin(metric_names)].copy()
        ground_truth_col = 'Ground Truth'
        estimated_speed_col = 'estimated speed'

        if ground_truth_col not in df.columns or estimated_speed_col not in df.columns:
            print(f"Error: Required columns '{ground_truth_col}' or '{estimated_speed_col}' not found in {file_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        df_filtered = df[df[estimated_speed_col] != 'data tidak ada'].copy()
        
        df_filtered[ground_truth_col] = pd.to_numeric(df_filtered[ground_truth_col], errors='coerce')
        df_filtered[estimated_speed_col] = pd.to_numeric(df_filtered[estimated_speed_col], errors='coerce')

        df_filtered.dropna(subset=[ground_truth_col, estimated_speed_col], inplace=True)

        if df_filtered.empty:
            print(f"No valid data for calculations in {file_path} after cleaning.")
            return None

        y_true = df_filtered[ground_truth_col]
        y_pred = df_filtered[estimated_speed_col]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = calculate_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics_data = {
            "Metric": ["MAE", "RMSE", "MAPE", "R2"],
            "Ground Truth": [mae, rmse, mape, r2] 
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_row = pd.DataFrame(columns=df.columns)
        for i, metric_name in enumerate(metrics_df["Metric"]):
            row_dict = {'Metric': metric_name, 'Ground Truth': metrics_df["Ground Truth"].iloc[i]}
            metrics_row = pd.concat([metrics_row, pd.DataFrame([row_dict])], ignore_index=True)

        updated_df = pd.concat([df, metrics_row], ignore_index=True)

        updated_df.to_csv(file_path, index=False)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2
        }
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    result_dir = "new_result"
    result_files = [f for f in os.listdir(result_dir) if f.endswith(".csv")]

    print("Calculating and adding metrics to each result file:")
    for file_name in result_files:
        file_path = os.path.join(result_dir, file_name)
        print(f"\n--- Processing {file_name} ---")
        metrics = calculate_metrics(file_path)
        if metrics:
            print(f"Metrics added to {file_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        else:
            print(f"Failed to add metrics to {file_name}.")

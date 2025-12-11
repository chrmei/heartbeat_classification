import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any


def generate_data_audit_report(
    data_dir: str = "data/original/", output_dir: str = "reports/data_audit/"
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to audit:")
    for file in csv_files:
        print(f"  - {file.name}")

    # Process each file
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        try:
            audit_report = create_audit_report_for_file(csv_file)
            save_audit_report(audit_report, csv_file.stem, output_dir)
            print(f"✓ Audit report generated for {csv_file.name}")
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {str(e)}")

    print(f"\nData audit reports completed! Check {output_dir} for results.")


def create_audit_report_for_file(file_path: Path) -> pd.DataFrame:
    # Load the data
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        raise Exception(f"Failed to load {file_path.name}: {str(e)}")

    n_rows, n_cols = df.shape
    audit_data = []

    # Analyze each column
    for col_idx in range(n_cols):
        column_data = df.iloc[:, col_idx]

        # Basic statistics
        missing_count = column_data.isnull().sum()
        missing_pct = (missing_count / len(column_data)) * 100
        dtype = str(column_data.dtype)
        unique_count = column_data.nunique()
        is_last_column = col_idx == n_cols - 1

        # Determine variable type
        if is_last_column:
            # For class labels
            try:
                numeric_values = pd.to_numeric(column_data, errors="coerce")
                if not numeric_values.isna().all():
                    unique_numeric = sorted(numeric_values.dropna().unique())
                    if len(unique_numeric) == 1:
                        var_type = "Categorical - Binary"
                    elif len(unique_numeric) == 2:
                        var_type = "Categorical - Binary"
                    elif 3 <= len(unique_numeric) <= 5:
                        var_type = "Categorical - 3 to 5 categories"
                    else:
                        var_type = "Quantitative"
                else:
                    var_type = (
                        "Categorical - Binary"
                        if unique_count <= 2
                        else "Categorical - 3 to 5 categories"
                    )
            except:
                var_type = (
                    "Categorical - Binary"
                    if unique_count <= 2
                    else "Categorical - 3 to 5 categories"
                )
        else:
            # For other columns
            if unique_count == 1:
                var_type = "Quantitative"
            elif unique_count == 2:
                var_type = "Categorical - Binary"
            elif 3 <= unique_count <= 5:
                var_type = "Categorical - 3 to 5 categories"
            else:
                var_type = "Quantitative"

        # Distribution information
        if var_type.startswith("Categorical"):
            unique_values = column_data.value_counts().head(10)
            distribution = ", ".join([f"{val}: {count}" for val, count in unique_values.items()])
        else:
            try:
                numeric_data = pd.to_numeric(column_data, errors="coerce")
                if not numeric_data.isna().all():
                    stats = {
                        "mean": numeric_data.mean(),
                        "std": numeric_data.std(),
                        "min": numeric_data.min(),
                        "max": numeric_data.max(),
                        "median": numeric_data.median(),
                    }
                    distribution = f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, Min: {stats['min']:.3f}, Max: {stats['max']:.3f}, Median: {stats['median']:.3f}"
                else:
                    distribution = "Non-numeric data"
            except:
                distribution = "Non-numeric data"

        # Determine if column should be deleted
        should_delete = "No"
        delete_reason = ""

        if missing_pct > 50:
            should_delete = "Consider"
            delete_reason = "High missing values (>50%)"
        elif unique_count == 1:
            should_delete = "No"  # set as none of these signals should be deleted
            delete_reason = ""
        elif unique_count == 0:
            should_delete = "Yes"
            delete_reason = "Empty column"

        # Generate description
        if col_idx == 0:
            description = "ECG signal amplitude values (normalized)"
        elif col_idx == n_cols - 1:
            # Analyze actual class labels
            try:
                numeric_values = pd.to_numeric(column_data, errors="coerce")
                if not numeric_values.isna().all():
                    unique_labels = sorted(numeric_values.dropna().unique())
                    if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
                        description = "Heartbeat class label (0=Normal, 1=Abnormal)"
                    elif len(unique_labels) <= 5:
                        label_desc = ", ".join([str(label) for label in unique_labels])
                        description = f"Heartbeat class label (values: {label_desc})"
                    else:
                        description = (
                            f"Heartbeat class label (numeric values: {len(unique_labels)} unique)"
                        )
                else:
                    description = (
                        f"Heartbeat class label (non-numeric, {unique_count} unique values)"
                    )
            except:
                description = f"Heartbeat class label ({unique_count} unique values)"
        else:
            description = f"ECG signal sample point {col_idx + 1}"

        # Add to audit data
        audit_data.append(
            {
                "# Column": col_idx + 1,
                "Name of the Column": f"col_{col_idx}",
                "Description": description,
                "Variable's type": dtype,
                "Percentage of missing values": f"{missing_pct:.1f}%",
                "Categorical / Quantitative": var_type,
                "Distribution": distribution,
                "Deleting?": should_delete,
                "Why Deleting?": delete_reason,
                "Comments": f"Unique values: {unique_count}",
            }
        )

    return pd.DataFrame(audit_data)


def save_audit_report(audit_df: pd.DataFrame, filename: str, output_dir: str) -> None:
    output_path = Path(output_dir) / f"data_audit_{filename}.csv"
    audit_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def get_data_summary(file_path: Path) -> Dict[str, Any]:
    try:
        df = pd.read_csv(file_path, header=None)

        summary = {
            "filename": file_path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.value_counts().to_dict(),
        }

        return summary

    except Exception as e:
        return {"filename": file_path.name, "error": str(e)}


def generate_summary_report(
    data_dir: str = "data/original/", output_file: str = "reports/data_audit/data_summary.txt"
) -> None:
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    # Ensure output directory exists
    os.makedirs(Path(output_file).parent, exist_ok=True)

    summaries = []
    for csv_file in csv_files:
        summary = get_data_summary(csv_file)
        summaries.append(summary)

    # Write summary report
    with open(output_file, "w") as f:
        f.write("DATA AUDIT SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        for summary in summaries:
            if "error" in summary:
                f.write(f"File: {summary['filename']}\n")
                f.write(f"Error: {summary['error']}\n\n")
            else:
                f.write(f"File: {summary['filename']}\n")
                f.write(f"  Rows: {summary['rows']:,}\n")
                f.write(f"  Columns: {summary['columns']}\n")
                f.write(f"  Memory Usage: {summary['memory_usage']:,} bytes\n")
                f.write(f"  Missing Values: {summary['missing_values']:,}\n")
                f.write(f"  Duplicate Rows: {summary['duplicate_rows']:,}\n")
                f.write(f"  Data Types: {summary['data_types']}\n\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("End of Summary Report\n")

    print(f"Summary report saved to: {output_file}")


if __name__ == "__main__":
    # Generate audit reports for all CSV files
    generate_data_audit_report()

    # Generate summary report
    generate_summary_report()

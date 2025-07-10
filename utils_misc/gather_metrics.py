import pandas as pd
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
import glob
import re
import argparse
from typing import List, Dict, Optional, Any

def load_metrics_files(directory_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Loads all *_metrics.json files from the specified directory.

    Args:
        directory_path: The path to the directory containing the metric files.

    Returns:
        A list of dictionaries, where each dictionary represents the data from a JSON file,
        or None if no files are found.
    """
    console = Console()
    
    pattern = str(directory_path / "*_metrics.json")
    metrics_files = glob.glob(pattern)
    
    if not metrics_files:
        console.print(f"[red]No *_metrics.json files found in {directory_path}[/red]")
        return None
    
    console.print(f"[green]Found {len(metrics_files)} metric files[/green]")
    
    all_metrics: List[Dict[str, Any]] = []
    
    for file_path in track(metrics_files, description="Loading files..."):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            file_name = Path(file_path).stem
            data['source_file'] = file_name
            all_metrics.append(data)
            
        except Exception as e:
            console.print(f"[red]Error loading {file_path}: {e}[/red]")
    
    return all_metrics

def create_dataframe(metrics_data: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    Converts metric data into a pandas DataFrame.

    Args:
        metrics_data: A list of dictionaries containing metrics.

    Returns:
        A pandas DataFrame with the normalized metric data, or None if the input is empty.
    """
    if not metrics_data:
        return None
    
    df = pd.json_normalize(metrics_data)
    return df

def extract_epoch_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly extracts epoch information from filenames.

    Args:
        df: The input DataFrame containing a 'source_file' column.

    Returns:
        A DataFrame with an added 'epoch' column, sorted by epoch.
    """
    df_copy = df.copy()
    
    epochs = df_copy['source_file'].str.extract(r'(\d+)_metrics', expand=False)
    
    df_copy['epoch'] = pd.to_numeric(epochs, errors='coerce')
    
    if df_copy['epoch'].isna().any():
        rprint("[yellow]Warning: Could not extract epoch for some files. Using row index as fallback.[/yellow]")
        df_copy['epoch'] = df_copy['epoch'].fillna(pd.Series(range(len(df_copy)), index=df_copy.index))

    return df_copy.sort_values('epoch')

def find_best_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Finds the best epoch for each metric.

    Args:
        df: The DataFrame containing the metrics over epochs.

    Returns:
        A DataFrame summarizing the best value and epoch for each metric, or None if the input is empty.
    """
    if df is None or df.empty:
        return None

    df_with_epochs = extract_epoch_info(df)

    numeric_columns = df_with_epochs.select_dtypes(include=['number']).columns
    metric_columns = [col for col in numeric_columns if col not in ['source_file', 'epoch']]

    best_metrics: List[Dict[str, Any]] = []
    for col in metric_columns:
        if col in df_with_epochs.columns and df_with_epochs[col].notna().any():
            col_lower = col.lower()
            
            minimize_metrics = ['rmse', 'mae', 'sam', 'error', 'mean se', 'mean ae', 'mean var', 'uce', 'auce', 'loss']
            maximize_metrics = ['psnr', 'ssim']

            is_loss_metric = any(term in col_lower for term in minimize_metrics)
            is_gain_metric = any(term in col_lower for term in maximize_metrics) if not is_loss_metric else False

            if is_loss_metric:
                best_idx = df_with_epochs[col].idxmin()
                metric_type = "Min (best)"
            elif is_gain_metric:
                best_idx = df_with_epochs[col].idxmax()
                metric_type = "Max (best)"
            else:
                default_to_min = any(term in col_lower for term in ['loss', 'error'])
                best_idx = df_with_epochs[col].idxmin() if default_to_min else df_with_epochs[col].idxmax()
                metric_type = "Min (best)" if default_to_min else "Max (best)"

            best_row = df_with_epochs.loc[best_idx]
            best_metrics.append({
                "Metric": col,
                "Best Value": best_row[col],
                "Epoch": int(best_row['epoch']) if pd.notna(best_row['epoch']) else 'N/A',
                "Type": metric_type
            })
            
    return pd.DataFrame(best_metrics)

def display_best_metrics_table(best_metrics_df: Optional[pd.DataFrame], console: Console) -> None:
    """
    Displays a table of the best metrics.

    Args:
        best_metrics_df: DataFrame with the best metrics to display.
        console: The Rich console object for printing.
    """
    if best_metrics_df is None or best_metrics_df.empty:
        console.print("[red]No metrics to display.[/red]")
        return

    table = Table(title="🏆 Best Performance by Metric", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Best Value", style="green", justify="right")
    table.add_column("Epoch", style="yellow", justify="right")
    table.add_column("Type", style="magenta")

    for _, row in best_metrics_df.iterrows():
        table.add_row(
            row["Metric"],
            f"{row['Best Value']:.4f}",
            str(row["Epoch"]),
            row["Type"]
        )
    
    console.print(table)

def save_best_metrics_to_json(best_metrics_df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves the best metrics DataFrame to a JSON file.

    Args:
        best_metrics_df: DataFrame with the best metrics.
        output_path: The path to the output JSON file.
    """
    if best_metrics_df is None or best_metrics_df.empty:
        return
        
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    best_metrics_df.to_json(output_path, orient="records", indent=4)
    
    console = Console()
    console.print(f"\n[green]Best metrics saved to: {output_path}[/green]")

def main() -> None:
    """
    Main function to load, process, and display metrics from a specified directory.
    """
    parser = argparse.ArgumentParser(description="Aggregate and find the best metrics from JSON files.")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="The target directory containing *_metrics.json files."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=False,
        help="The path to the output JSON file. If not provided, 'best_metrics.json' will be saved in the target directory."
    )
    args = parser.parse_args()
    
    console = Console()
    target_directory = Path(args.directory)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = target_directory / "best_metrics.json"

    console.print(f"[blue]Analyzing directory: {target_directory}[/blue]")
    
    if not target_directory.exists():
        console.print(f"[red]Directory {target_directory} does not exist.[/red]")
        return
    
    metrics_data = load_metrics_files(target_directory)
    if not metrics_data:
        return
    
    df = create_dataframe(metrics_data)
    if df is None:
        console.print("[red]Failed to create DataFrame.[/red]")
        return

    best_metrics_df = find_best_metrics(df)

    display_best_metrics_table(best_metrics_df, console)

    save_best_metrics_to_json(best_metrics_df, output_path)

    console.print(f"\n[green]Analysis finished successfully![/green]")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# convert_to_excel.py

import os
import pandas as pd
import argparse

def convert_csv_to_excel(csv_path, excel_path=None):
    """
    Convert a CSV file to Excel format.
    
    Args:
        csv_path: Path to the CSV file
        excel_path: Path to the output Excel file (optional)
    """
    if excel_path is None:
        # Use the same path but with .xlsx extension
        excel_path = os.path.splitext(csv_path)[0] + '.xlsx'
    
    print(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Converting to Excel and saving to {excel_path}")
    df.to_excel(excel_path, index=False)
    
    print(f"Successfully converted {csv_path} to {excel_path}")
    return excel_path

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Convert CSV to Excel")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--excel", type=str, help="Path to the output Excel file (optional)")
    args = parser.parse_args()
    
    convert_csv_to_excel(args.csv, args.excel)

if __name__ == "__main__":
    main()

import pandas as pd

# function to check dataframe if columns contain blank entries
def check_dataframe_for_blanks(df: pd.DataFrame) -> None:
    print("Checking dataframe for blank entries for all columns...")
    blank_mask_all = df.isna() | (df.astype(str).apply(lambda x: x.str.strip() == ''))
    print(blank_mask_all.any())  # True/False per column

# function to check blank entries based on dataframe column
def check_column_contains_blanks(column: pd.Series) -> None:
    count_blank = column.apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')).sum()
    print("\nChecking column", column.name, "for blank entries...")
    print("Blank entries found:", count_blank)
    print("Total entries found:", column.size)

# function to export dataframe to csv
def export_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool = False, encoding: str = "utf-8-sig") -> None:
    """
    Export a DataFrame to a CSV file.

    Parameters:
    - df: pd.DataFrame - The DataFrame to export.
    - file_path: str - The path to the output CSV file.
    - index: bool - Whether to write row indices. Default is False.
    - encoding: str - The encoding for the output file. Default is "utf-8-sig".
    """
    try:
        print("Exporting dataframe to csv file:", file_path)
        df.to_csv(
            file_path,
            index=index,
            encoding=encoding
        )
        print(f"DataFrame successfully exported to '{file_path}'")
    except Exception as e:
        print(f"Error exporting DataFrame: {e}")
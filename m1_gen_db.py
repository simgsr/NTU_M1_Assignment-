"""
Job Postings ETL Pipeline

This script processes a large CSV file with nested JSON fields containing category objects,
normalizes the categories column, and persists the data to DuckDB and Pickle formats.
"""

import pandas as pd
import duckdb
import json
import os
import numpy as np

# ============================================================================
# 1. User Input Handling
# ============================================================================

def get_file_path():
    """Prompt user for CSV file path."""
    file_path = input("Enter the path to the CSV file: ").strip()
    file_path = file_path.strip('"').strip("'")
    return file_path

# ============================================================================
# 2. Data Loading Function
# ============================================================================

def load_csv_efficiently(file_path):
    """Load CSV with optimized dtypes for memory efficiency."""
    print("Loading CSV...")
    
    # First pass: read dtypes from sample
    sample = pd.read_csv(file_path, nrows=1000)
    
    dtype_dict = {}
    for col in sample.columns:
        if col == 'categories':
            dtype_dict[col] = 'object'
        elif sample[col].dtype == 'object':
            if sample[col].nunique() / len(sample) < 0.5:
                dtype_dict[col] = 'category'
            else:
                dtype_dict[col] = 'object'
        elif sample[col].dtype == 'int64':
            dtype_dict[col] = 'int32'
        elif sample[col].dtype == 'float64':
            dtype_dict[col] = 'float32'
    
    # Read full CSV with optimized dtypes
    df = pd.read_csv(file_path, dtype=dtype_dict)
    return df

# ============================================================================
# 3. Category Normalization Function
# ============================================================================

def parse_categories(json_str):
    """
    Parse JSON string to extract category id and name.
    Expected format: [{"id": 13, "category": "Environment / Health"}, ...]
    """
    try:
        if pd.isna(json_str):
            return []
        
        data = json.loads(json_str)
        
        if isinstance(data, list):
            categories = []
            for item in data:
                if isinstance(item, dict):
                    # Extract id and category fields
                    cat_id = item.get('id')
                    cat_name = item.get('category')
                    
                    if cat_id is not None and cat_name is not None:
                        categories.append({
                            'category_id': int(cat_id),  # Keep as int for efficiency
                            'category_name': str(cat_name).strip()
                        })
            return categories
        return []
    except:
        return []

def normalize_categories(df, job_id_col='metadata_jobPostId'):
    """Normalize the categories column into lookup and junction tables."""
    
    # Create jobs table (drop categories column)
    jobs_df = df.drop(columns=['categories'])
    
    # Parse all categories
    print("Parsing categories...")
    all_categories = []
    job_category_links = []
    
    total_rows = len(df)
    for idx, row in df.iterrows():
        job_id = row[job_id_col]
        categories = parse_categories(row['categories'])
        
        for cat in categories:
            # Store for categories lookup
            all_categories.append((cat['category_id'], cat['category_name']))
            # Store for junction table
            job_category_links.append((job_id, cat['category_id']))
        
        # Progress indicator (every 500k rows)
        if idx > 0 and idx % 500000 == 0:
            print(f"  Processed {idx:,} rows...")
    
    # Create categories lookup table
    print("Creating categories lookup table...")
    unique_categories = list(set(all_categories))  # Remove duplicates
    categories_df = pd.DataFrame(unique_categories, columns=['category_id', 'category_name'])
    categories_df = categories_df.sort_values('category_id').reset_index(drop=True)
    
    # Create job_categories junction table
    print("Creating job-categories junction table...")
    job_categories_df = pd.DataFrame(job_category_links, columns=['job_id', 'category_id'])
    job_categories_df = job_categories_df.drop_duplicates().reset_index(drop=True)
    
    # Optimize dtypes
    categories_df['category_id'] = categories_df['category_id'].astype('int32')
    categories_df['category_name'] = categories_df['category_name'].astype('category')
    
    job_categories_df['job_id'] = job_categories_df['job_id'].astype('category')
    job_categories_df['category_id'] = job_categories_df['category_id'].astype('int32')
    
    return jobs_df, categories_df, job_categories_df

def load_and_normalize():
    """Main function to load and normalize data."""
    file_path = get_file_path()
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None, None, None
    
    df = load_csv_efficiently(file_path)
    
    if 'categories' not in df.columns:
        print("Error: 'categories' column not found in CSV")
        return None, None, None, None
    
    jobs_df, categories_df, job_categories_df = normalize_categories(df)
    
    output_dir = os.path.dirname(file_path)
    return jobs_df, categories_df, job_categories_df, output_dir

# ============================================================================
# 4. Output Persistence Functions
# ============================================================================

def save_to_duckdb(jobs_df, categories_df, job_categories_df, output_dir):
    """Save DataFrames to DuckDB database."""
    db_path = os.path.join(output_dir, 'job_postings.db')
    conn = duckdb.connect(db_path)
    
    # Create tables
    conn.execute("CREATE OR REPLACE TABLE jobs AS SELECT * FROM jobs_df")
    conn.execute("CREATE OR REPLACE TABLE categories AS SELECT * FROM categories_df")
    conn.execute("CREATE OR REPLACE TABLE job_categories AS SELECT * FROM job_categories_df")
    
    # Create indexes
    if len(jobs_df) > 0:
        conn.execute("CREATE INDEX idx_job_id ON jobs(metadata_jobPostId)")
    if len(job_categories_df) > 0:
        conn.execute("CREATE INDEX idx_jc_job_id ON job_categories(job_id)")
        conn.execute("CREATE INDEX idx_jc_category_id ON job_categories(category_id)")
    
    conn.close()

def save_to_pickle(jobs_df, categories_df, job_categories_df, output_dir):
    """Save DataFrames to Pickle files."""
    base_path = os.path.join(output_dir, 'job_postings')
    
    jobs_df.to_pickle(f'{base_path}_jobs.pkl', compression=None)
    categories_df.to_pickle(f'{base_path}_categories.pkl', compression=None)
    job_categories_df.to_pickle(f'{base_path}_job_categories.pkl', compression=None)

# ============================================================================
# 5. Data Validation Display Function
# ============================================================================

def display_validation_info(jobs_df, categories_df, job_categories_df):
    """Display only the required validation information."""
    
    print("\n=== DATA LOADED ===")
    
    # Display shapes
    print(f"Jobs DataFrame: Shape = {jobs_df.shape}")
    print(f"Categories DataFrame: Shape = {categories_df.shape}")
    print(f"JobCategories DataFrame: Shape = {job_categories_df.shape}")
    
    # Display Jobs columns (first 10 only)
    cols_list = jobs_df.columns.tolist()
    cols_str = str(cols_list[:10]) + ('...' if len(cols_list) > 10 else '')
    print(f"Jobs columns: {cols_str}")
    print()
    
    # Display unique counts for Jobs table
    print("Jobs unique value counts per column:")
    for col in jobs_df.columns:
        try:
            unique_count = jobs_df[col].nunique()
            print(f"  {col}: {unique_count}")
        except:
            print(f"  {col}: Error")
    print()
    
    # Display unique counts for Categories table
    if len(categories_df) > 0:
        print("Categories unique value counts per column:")
        for col in categories_df.columns:
            try:
                unique_count = categories_df[col].nunique()
                print(f"  {col}: {unique_count}")
            except:
                print(f"  {col}: Error")
    else:
        print("Categories table is empty")
    print()
    
    # Display unique counts for JobCategories table
    if len(job_categories_df) > 0:
        print("JobCategories unique value counts per column:")
        for col in job_categories_df.columns:
            try:
                unique_count = job_categories_df[col].nunique()
                print(f"  {col}: {unique_count}")
            except:
                print(f"  {col}: Error")
    else:
        print("JobCategories table is empty")

# ============================================================================
# 6. Main Execution
# ============================================================================

def main():
    """Main ETL pipeline execution."""
    
    jobs_df, categories_df, job_categories_df, output_dir = load_and_normalize()
    
    if jobs_df is None:
        print("Failed to load data. Exiting.")
        return
    
    display_validation_info(jobs_df, categories_df, job_categories_df)
    
    print("\nSaving to DuckDB...")
    save_to_duckdb(jobs_df, categories_df, job_categories_df, output_dir)
    
    print("Saving to Pickle...")
    save_to_pickle(jobs_df, categories_df, job_categories_df, output_dir)
    
    print("\nETL pipeline completed successfully!")

if __name__ == "__main__":
    main()
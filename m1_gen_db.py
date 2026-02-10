import os
import duckdb
import pandas as pd
import ast

def preprocess_and_normalize(csv_path, 
                            direct_db_name='SGJobData.db', 
                            normalized_db_name='SGJobData_Normalized.db'):
    """
    Preprocess CSV containing job data with nested categories, create a direct database,
    and normalize into 3NF schema in a separate database.
    
    Args:
        csv_path: Path to input CSV file
        direct_db_name: Name for the direct (denormalized) database file
        normalized_db_name: Name for the normalized (3NF) database file
    """
    try:
        # === STEP 1: READ AND INITIAL PROCESSING ===
        print("Reading CSV file...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df):,} rows from CSV")
        
        # Process categories column (stringified list of dicts)
        normalized_rows = []
        
        # === STEP 2: UNNEST CATEGORIES COLUMN ===
        # The categories column contains stringified lists of dictionaries
        # We need to parse and flatten this structure
        for idx, row in df.iterrows():
            categories_str = row['categories']
            job_id = row['metadata_jobPostId']
            
            try:
                if isinstance(categories_str, str):
                    # Safely parse string representation of list using ast.literal_eval
                    # This converts the string to actual Python list of dictionaries
                    categories_list = ast.literal_eval(categories_str.strip())
                    
                    if categories_list:
                        # Create one row per category (unnesting/flattening)
                        for cat in categories_list:
                            new_row = dict(row)  # Copy all original columns
                            new_row['Cat_ID'] = cat.get('id')  # Extract category ID
                            new_row['Cat_Name'] = cat.get('category')  # Extract category name
                            normalized_rows.append(new_row)
                    else:
                        # Job has no categories (empty list)
                        new_row = dict(row)
                        new_row['Cat_ID'] = None
                        new_row['Cat_Name'] = None
                        normalized_rows.append(new_row)
                else:
                    # Handle null values or already parsed values
                    new_row = dict(row)
                    new_row['Cat_ID'] = None
                    new_row['Cat_Name'] = None
                    normalized_rows.append(new_row)
                    
            except (SyntaxError, ValueError):
                # Fallback parsing for malformed data (in case ast.literal_eval fails)
                if isinstance(categories_str, str) and 'id' in categories_str:
                    import re
                    # Use regex to extract category IDs and names
                    ids = re.findall(r'"id":(\d+)', categories_str)
                    names = re.findall(r'"category":"([^"]+)"', categories_str)
                    
                    for cat_id, cat_name in zip(ids, names):
                        new_row = dict(row)
                        new_row['Cat_ID'] = int(cat_id)
                        new_row['Cat_Name'] = cat_name
                        normalized_rows.append(new_row)
                else:
                    new_row = dict(row)
                    new_row['Cat_ID'] = None
                    new_row['Cat_Name'] = None
                    normalized_rows.append(new_row)
        
        # === STEP 3: CREATE NORMALIZED DATAFRAME ===
        normalized_df = pd.DataFrame(normalized_rows)
        
        # Remove original nested column as we've extracted its contents
        if 'categories' in normalized_df.columns:
            normalized_df = normalized_df.drop('categories', axis=1)
        
        print(f"Created {len(normalized_df):,} normalized rows")
        
        # === STEP 4: CREATE DIRECT (DENORMALIZED) DATABASE ===
        # This database contains the flattened but still denormalized data
        print(f"\nCreating direct database: {direct_db_name}")
        direct_con = duckdb.connect(direct_db_name)
        direct_con.register('normalized_data', normalized_df)
        direct_con.execute("CREATE TABLE SGJobData AS SELECT * FROM normalized_data")
        
        # Report direct database statistics
        direct_count = direct_con.execute("SELECT COUNT(*) FROM SGJobData").fetchone()[0]
        print(f"   - Direct database created with {direct_count:,} rows")
        direct_con.close()
        
        # === STEP 5: CREATE NORMALIZED (3NF) DATABASE ===
        print(f"\nCreating normalized database: {normalized_db_name}")
        norm_con = duckdb.connect(normalized_db_name)
        norm_con.register('normalized_data', normalized_df)
        
        # Create normalized schema (3NF) with three tables:
        
        # 1. CATEGORIES DIMENSION TABLE
        # Contains unique category entities with their attributes
        # This eliminates duplicate category information
        norm_con.execute("""
            CREATE TABLE Categories AS
            SELECT DISTINCT 
                Cat_ID,
                Cat_Name
            FROM normalized_data
            WHERE Cat_ID IS NOT NULL
            ORDER BY Cat_ID
        """)
        
        # 2. JOBS FACT TABLE
        # Contains core job information without repeating categories
        # This table has one row per job with all job-specific attributes
        norm_con.execute("""
            CREATE TABLE Jobs AS
            SELECT DISTINCT
                metadata_jobPostId,
                employmentTypes,
                metadata_expiryDate,
                metadata_isPostedOnBehalf,
                metadata_newPostingDate,
                metadata_originalPostingDate,
                metadata_repostCount,
                metadata_totalNumberJobApplication,
                metadata_totalNumberOfView,
                minimumYearsExperience,
                numberOfVacancies,
                positionLevels,
                postedCompany_name,
                salary_maximum,
                salary_minimum,
                salary_type,
                status_jobStatus,
                title,
                average_salary
            FROM normalized_data
        """)
        
        # 3. JOBCATEGORIES JUNCTION TABLE
        # Resolves many-to-many relationship between Jobs and Categories
        # A job can have multiple categories, and a category can belong to multiple jobs
        norm_con.execute("""
            CREATE TABLE JobCategories AS
            SELECT DISTINCT
                metadata_jobPostId,
                Cat_ID
            FROM normalized_data
            WHERE Cat_ID IS NOT NULL
        """)
        
        # === STEP 6: REPORT FINAL STATISTICS ===
        print("\nNormalized database created successfully:")
        print(f"   - Categories: {norm_con.execute('SELECT COUNT(*) FROM Categories').fetchone()[0]} unique categories")
        print(f"   - Jobs: {norm_con.execute('SELECT COUNT(*) FROM Jobs').fetchone()[0]} distinct jobs")
        print(f"   - JobCategories: {norm_con.execute('SELECT COUNT(*) FROM JobCategories').fetchone()[0]} relationships")
        
        norm_con.close()
        return True
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to handle user input and initiate processing.
    """
    while True:
        csv_path = input("\nEnter path to CSV file (or 'quit' to exit): ").strip()
        
        if csv_path.lower() == 'quit':
            break
        
        # Clean up path (remove quotes if user drag-and-dropped file)
        csv_path = csv_path.strip('"').strip("'")
        
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        
        # Process the CSV file and create both databases
        preprocess_and_normalize(csv_path)
        break

if __name__ == "__main__":
    main()
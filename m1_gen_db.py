import os
import duckdb
import pandas as pd
import ast

def preprocess_and_normalize(csv_path, 
                            direct_db_name='db/SGJobData.db', 
                            normalized_db_name='db/SGJobData_Normalized.db'):
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
        
        # map categories to generic sectors
        sector_map = {                        
                        'Accounting / Auditing / Taxation': 'Financial & Professional Services',
                        'Admin / Secretarial': 'Business Support & Administration',
                        'Advertising / Media': 'Creative, Media & Design',
                        'Architecture / Interior Design': 'Built Environment & Real Estate',
                        'Banking and Finance': 'Financial & Professional Services',
                        'Building and Construction': 'Built Environment & Real Estate',
                        'Consulting': 'Financial & Professional Services',
                        'Customer Service': 'Sales, Retail & Personal Services',
                        'Design': 'Creative, Media & Design',
                        'Education and Training': 'Public & Social Services',
                        'Engineering': 'Engineering & Manufacturing',
                        'Entertainment': 'Creative, Media & Design',
                        'Environment / Health': 'Public & Social Services',
                        'Events / Promotions': 'Sales, Retail & Personal Services',
                        'F&B': 'Hospitality & Tourism',
                        'General Management': 'Business Support & Administration',
                        'General Work': 'Others & General Work',
                        'Healthcare / Pharmaceutical': 'Healthcare & Life Sciences',
                        'Hospitality': 'Hospitality & Tourism',
                        'Human Resources': 'Business Support & Administration',
                        'Information Technology': 'Technology & Telecommunications',
                        'Insurance': 'Financial & Professional Services',
                        'Legal': 'Financial & Professional Services',
                        'Logistics / Supply Chain': 'Logistics, Trade & Supply Chain',
                        'Manufacturing': 'Engineering & Manufacturing',
                        'Marketing / Public Relations': 'Creative, Media & Design',
                        'Medical / Therapy Services': 'Healthcare & Life Sciences',
                        'Others': 'Others & General Work',
                        'Personal Care / Beauty': 'Sales, Retail & Personal Services',
                        'Precision Engineering': 'Engineering & Manufacturing',
                        'Professional Services': 'Financial & Professional Services',
                        'Public / Civil Service': 'Public & Social Services',
                        'Purchasing / Merchandising': 'Logistics, Trade & Supply Chain',
                        'Real Estate / Property Management': 'Built Environment & Real Estate',
                        'Repair and Maintenance': 'Engineering & Manufacturing',
                        'Risk Management': 'Financial & Professional Services',
                        'Sales / Retail': 'Sales, Retail & Personal Services',
                        'Sciences / Laboratory / R&D': 'Healthcare & Life Sciences',
                        'Security and Investigation': 'Public & Social Services',
                        'Social Services': 'Public & Social Services',
                        'Telecommunications': 'Technology & Telecommunications',
                        'Travel / Tourism': 'Hospitality & Tourism',
                        'Wholesale Trade': 'Logistics, Trade & Supply Chain'
                    }
        normalized_df['Sector'] = normalized_df['Cat_Name'].map(sector_map)

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
            CREATE TABLE Categories (
                Cat_ID INT PRIMARY KEY,
                Cat_Name VARCHAR NOT NULL,      
                Sector VARCHAR NOT NULL         
            );
        """) 
              
        norm_con.execute("""
            INSERT INTO Categories
            SELECT DISTINCT Cat_ID,
                            Cat_Name,
                            Sector
            FROM normalized_data
            WHERE Cat_ID IS NOT NULL
            ORDER BY Cat_ID;
        """)
        
        # 2. JOBS FACT TABLE
        # Contains core job information without repeating categories
        # This table has one row per job with all job-specific attributes
        norm_con.execute("""
            CREATE TABLE Jobs (
                metadata_jobPostId VARCHAR PRIMARY KEY,
                employmentTypes VARCHAR,
                metadata_expiryDate DATE,
                metadata_isPostedOnBehalf BOOLEAN,
                metadata_newPostingDate DATE,
                metadata_originalPostingDate DATE,
                metadata_repostCount INT,
                metadata_totalNumberJobApplication INT,
                metadata_totalNumberOfView INT,
                minimumYearsExperience INT,
                numberOfVacancies INT,
                positionLevels VARCHAR,
                postedCompany_name VARCHAR,
                salary_maximum DECIMAL,
                salary_minimum DECIMAL,
                status_jobStatus VARCHAR,
                title VARCHAR,
                average_salary DECIMAL
            );
        """)       
        
        norm_con.execute("""
            INSERT INTO Jobs
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
                status_jobStatus,
                title,
                average_salary
            FROM normalized_data;
        """)
        
        # 3. JOBCATEGORIES JUNCTION TABLE
        # Resolves many-to-many relationship between Jobs and Categories
        # A job can have multiple categories, and a category can belong to multiple jobs
        norm_con.execute("""
            CREATE TABLE JobCategories (
                metadata_jobPostId VARCHAR NOT NULL,
                Cat_ID INT NOT NULL,      
                CONSTRAINT jobcat_pk PRIMARY KEY (metadata_jobPostId, Cat_ID),
                CONSTRAINT jobcat_fk1 FOREIGN KEY (metadata_jobPostId) REFERENCES Jobs (metadata_jobPostId),
                CONSTRAINT jobcat_fk2 FOREIGN KEY (Cat_ID) REFERENCES Categories (Cat_ID)
            );
        """) 
        
        norm_con.execute("""
            INSERT INTO JobCategories
            SELECT DISTINCT
                metadata_jobPostId,
                Cat_ID
            FROM normalized_data
            WHERE Cat_ID IS NOT NULL;
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
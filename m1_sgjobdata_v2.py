import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import os

# ============================================================================
# LOAD DATA
# ============================================================================
def load_data(db_path):
    """
    Load data from DuckDB database.
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        Tuple of (jobs_df, categories_df, jobcategories_df)
    """
    print("Loading data from database...")
    
    # Connect to the database using DuckDB
    conn = duckdb.connect(db_path)
    
    # Load each table
    jobs_df = conn.execute("SELECT * FROM Jobs").fetchdf()
    categories_df = conn.execute("SELECT * FROM Categories").fetchdf()
    jobcategories_df = conn.execute("SELECT * FROM JobCategories").fetchdf()
    
    # Close the connection
    conn.close()
    
    print("\n=== DATA LOADED ===")
    print(f"Jobs DataFrame: Shape = {jobs_df.shape}")
    print(f"Categories DataFrame: Shape = {categories_df.shape}")
    print(f"JobCategories DataFrame: Shape = {jobcategories_df.shape}")
    print(f"Jobs columns: {list(jobs_df.columns)}")
    
    return jobs_df, categories_df, jobcategories_df

# ============================================================================
# CLEANING AND PREPARING DATA
# ============================================================================
def clean_and_prepare_data(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the jobs dataframe by removing redundant columns
    and converting data types.
    
    Args:
        jobs_df: Raw jobs dataframe
        
    Returns:
        Cleaned jobs dataframe
    """
    print("\n=== CLEANING AND PREPARING DATA ===")
    
    # Create a copy to avoid modifying original
    filtered_jobs = jobs_df.copy()
    
    # Remove redundant columns - keep only what we need for analysis
    columns_to_drop = [
        'occupationId', 'status_id', 'salary_type', 
        'salary_maximum', 'salary_minimum', 
        'employmentTypes', 'metadata_expiryDate', 
        'metadata_isPostedOnBehalf', 'metadata_newPostingDate', 
        'metadata_originalPostingDate', 'metadata_repostCount'
        'title', 'postedCompany_name',  'status_jobStatus', 
        'numberOfVacancies', 'metadata_totalNumberJobApplication', 
        'metadata_totalNumberOfView'
    ]
    
    # Only drop columns that exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in filtered_jobs.columns]
    if existing_columns_to_drop:
        filtered_jobs = filtered_jobs.drop(columns=existing_columns_to_drop)
        print(f"Dropped {len(existing_columns_to_drop)} redundant columns. New shape: {filtered_jobs.shape}")
        print(f"Remaining columns: {list(filtered_jobs.columns)}")
    
    # Convert columns to appropriate data types
    numeric_cols = ['minimumYearsExperience', 'average_salary']
    for col in numeric_cols:
        if col in filtered_jobs.columns:
            filtered_jobs[col] = pd.to_numeric(filtered_jobs[col], errors='coerce')
            missing = filtered_jobs[col].isna().sum()
            print(f"Converted {col} to numeric. Missing values: {missing:,} ({missing/len(filtered_jobs)*100:.1f}%)")
    
    # Clean company names: convert to title case
    if 'postedCompany_name' in filtered_jobs.columns:
        filtered_jobs['postedCompany_name'] = filtered_jobs['postedCompany_name'].str.title()
        print("Converted company names to title case")
    
    return filtered_jobs


# ============================================================================
# CREATE EXPERIENCE LEVELS
# ============================================================================
def create_experience_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create experience level categories from minimumYearsExperience.
    
    Args:
        df: DataFrame with minimumYearsExperience column
        
    Returns:
        DataFrame with new 'experience_level' column
    """
    print("\n=== CREATING EXPERIENCE LEVEL CATEGORIES ===")
    
    # Define bins and labels for experience levels in 5-year intervals
    bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, np.inf]
    labels = [
        '0-5 years',
        '6-10 years',
        '11-15 years',
        '16-20 years',
        '21-25 years',
        '26-30 years',
        '31-35 years',
        '36-40 years',
        '40+ years'
    ]
    
    # Create the new column 'experience_level' using pandas.cut
    df['experience_level'] = pd.cut(
        df['minimumYearsExperience'], 
        bins=bins, 
        labels=labels, 
        right=True
    )
    
    # Display distribution
    exp_counts = df['experience_level'].value_counts().sort_index()
    print("Experience level distribution:")
    for level, count in exp_counts.items():
        pct = count / len(df) * 100
        print(f"  {level}: {count:>8,} jobs ({pct:>5.1f}%)")
    
    return df


# ============================================================================
# CREATE INDUSTRY CATEGORIES
# ============================================================================
def create_industry_categories(categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create broader industry categories from detailed job categories.
    
    Args:
        categories_df: DataFrame with detailed job categories
        
    Returns:
        DataFrame with new 'Industry_Category' column
    """
    print("\n=== CREATING INDUSTRY CATEGORIES ===")
    
    industry_mapping = {
        # Business & Professional Services
        'Accounting / Auditing / Taxation': 'Business & Professional Services',
        'Admin / Secretarial': 'Business & Professional Services',
        'Banking and Finance': 'Business & Professional Services',
        'Consulting': 'Business & Professional Services',
        'Insurance': 'Business & Professional Services',
        'Legal': 'Business & Professional Services',
        'Professional Services': 'Business & Professional Services',
        'Risk Management': 'Business & Professional Services',
        
        # Engineering & Technical
        'Engineering': 'Engineering & Technical',
        'Information Technology': 'Engineering & Technical',
        'Precision Engineering': 'Engineering & Technical',
        'Repair and Maintenance': 'Engineering & Technical',
        'Telecommunications': 'Engineering & Technical',
        
        # Sales, Marketing & Retail
        'Advertising / Media': 'Sales, Marketing & Retail',
        'Events / Promotions': 'Sales, Marketing & Retail',
        'Marketing / Public Relations': 'Sales, Marketing & Retail',
        'Sales / Retail': 'Sales, Marketing & Retail',
        'Wholesale Trade': 'Sales, Marketing & Retail',
        
        # Healthcare & Life Sciences
        'Healthcare / Pharmaceutical': 'Healthcare & Life Sciences',
        'Medical / Therapy Services': 'Healthcare & Life Sciences',
        'Environment / Health': 'Healthcare & Life Sciences',
        
        # Hospitality & Services
        'Customer Service': 'Hospitality & Services',
        'F&B': 'Hospitality & Services',
        'Hospitality': 'Hospitality & Services',
        'Personal Care / Beauty': 'Hospitality & Services',
        'Travel / Tourism': 'Hospitality & Services',
        
        # Creative & Design
        'Architecture / Interior Design': 'Creative & Design',
        'Design': 'Creative & Design',
        'Entertainment': 'Creative & Design',
        
        # Manufacturing & Logistics
        'Logistics / Supply Chain': 'Manufacturing & Logistics',
        'Manufacturing': 'Manufacturing & Logistics',
        'Purchasing / Merchandising': 'Manufacturing & Logistics',
        
        # Construction & Real Estate
        'Building and Construction': 'Construction & Real Estate',
        'Real Estate / Property Management': 'Construction & Real Estate',
        
        # Management & HR
        'General Management': 'Management & HR',
        'Human Resources': 'Management & HR',
        
        # Education & Training
        'Education and Training': 'Education & Training',
        
        # Research & Development
        'Sciences / Laboratory / R&D': 'Research & Development',
        
        # Public Sector & Social Services
        'Public / Civil Service': 'Public Sector & Social Services',
        'Social Services': 'Public Sector & Social Services',
        
        # General & Miscellaneous
        'General Work': 'General & Support Services',
        'Security and Investigation': 'General & Support Services',
        'Others': 'Other / Miscellaneous'
    }
    
    # Apply mapping to categories dataframe
    categories_df['Industry_Category'] = categories_df['Cat_Name'].map(industry_mapping)
    
    # Handle any unmapped categories
    unmapped = categories_df[categories_df['Industry_Category'].isna()]
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} categories not mapped to industry")
        categories_df['Industry_Category'] = categories_df['Industry_Category'].fillna('Other / Miscellaneous')
    
    # Count unique industry categories
    n_unique = categories_df['Industry_Category'].nunique()
    print(f"Created {n_unique} industry categories")
    
    print("\nIndustry distribution:")
    industry_counts = categories_df['Industry_Category'].value_counts()
    for industry, count in industry_counts.items():
        print(f"  {industry}: {count} sub-categories")
    
    return categories_df


# ============================================================================
# JOIN DATASETS
# ============================================================================
def join_datasets(jobs_df: pd.DataFrame, 
                  categories_df: pd.DataFrame, 
                  jobcategories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join all three datasets into one comprehensive dataframe.
    
    Args:
        jobs_df: Jobs dataframe
        categories_df: Categories dataframe with industry mapping
        jobcategories_df: Job-Categories mapping dataframe
        
    Returns:
        Joined dataframe
    """
    print("\n=== JOINING DATASETS ===")
    
    # First, join JobCategories with Categories to get Industry_Category for each job
    job_with_categories = pd.merge(
        jobcategories_df, 
        categories_df[['Cat_ID', 'Industry_Category']], 
        on='Cat_ID', 
        how='left'
    )
    
    # For jobs with multiple categories, take the first category
    job_industry = job_with_categories.groupby('metadata_jobPostId').first().reset_index()
    
    # Now join with the jobs dataframe - only keep Industry_Category from the join
    final_df = pd.merge(
        jobs_df, 
        job_industry[['metadata_jobPostId', 'Industry_Category']], 
        on='metadata_jobPostId', 
        how='left'
    )
    
    print(f"Final merged dataframe shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    
    # Check for missing values
    print("\nMissing values:")
    for col in ['Industry_Category', 'average_salary', 'minimumYearsExperience']:
        if col in final_df.columns:
            missing = final_df[col].isna().sum()
            percent = missing / len(final_df) * 100
            print(f"  {col}: {missing:>8,} ({percent:>5.1f}%)")
    
    return final_df


# ============================================================================
# APPLY REALISTIC SALARY FILTERS
# ============================================================================
def apply_realistic_salary_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply realistic salary filters based on Singapore market rates and experience levels.
    
    Args:
        df: DataFrame containing salary and experience data
        
    Returns:
        Filtered DataFrame with realistic salary ranges
    """
    print("\n=== APPLYING REALISTIC SALARY FILTERS ===")
    print(f"Initial data shape: {df.shape}")
    
    # Make a copy
    filtered_df = df.copy()
    
    # Step 1: Basic sanity checks
    # Remove salaries below reasonable minimum (SGD 1,000/month)
    min_reasonable = 1000
    mask_min = (filtered_df['average_salary'] >= min_reasonable)
    filtered_df = filtered_df[mask_min]
    print(f"After removing salaries < ${min_reasonable:,}: {filtered_df.shape}")
    
    # Remove salaries above reasonable maximum (SGD 50,000/month)
    max_reasonable = 50000
    mask_max = (filtered_df['average_salary'] <= max_reasonable)
    filtered_df = filtered_df[mask_max]
    print(f"After removing salaries > ${max_reasonable:,}: {filtered_df.shape}")
    
    # Remove rows with missing experience level
    if 'experience_level' in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=['experience_level'])
        print(f"After removing missing experience levels: {filtered_df.shape}")
    
    # Step 2: Experience-level specific filtering
    print("\nApplying experience-level specific filters:")
    
    # Define reasonable ranges by experience level (based on Singapore market)
    salary_ranges = {
        '0-5 years': {'min': 2000, 'max': 12000},
        '6-10 years': {'min': 3500, 'max': 20000},
        '11-15 years': {'min': 5000, 'max': 30000},
        '16-20 years': {'min': 5000, 'max': 50000},
        '21-25 years': {'min': 5000, 'max': 50000},
        '26-30 years': {'min': 5000, 'max': 50000},
        '31-35 years': {'min': 5000, 'max': 50000},
        '36-40 years': {'min': 5000, 'max': 50000},
        '40+ years': {'min': 5000, 'max': 50000}
    }
    
    # Create mask for each experience level
    masks = []
    for exp_level, ranges in salary_ranges.items():
        # For this experience level, keep if within range
        mask = (
            (filtered_df['experience_level'] == exp_level) & 
            (filtered_df['average_salary'] >= ranges['min']) & 
            (filtered_df['average_salary'] <= ranges['max'])
        ) | (filtered_df['experience_level'] != exp_level)
        
        # Count removals for this level
        level_data = filtered_df[filtered_df['experience_level'] == exp_level]
        if len(level_data) > 0:
            removed = level_data[
                (level_data['average_salary'] < ranges['min']) | 
                (level_data['average_salary'] > ranges['max'])
            ].shape[0]
            
            if removed > 0:
                total_level = len(level_data)
                percent_removed = removed / total_level * 100
                print(f"  {exp_level}: Removed {removed:>4,}/{total_level:>6,} ({percent_removed:>5.1f}%)")
        
        masks.append(mask)
    
    # Combine all masks
    if masks:
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask & mask
        filtered_df = filtered_df[final_mask]
    
    print(f"\nFinal shape after realistic filters: {filtered_df.shape}")
    
    return filtered_df


# ============================================================================
# APPLY IQR OUTLIER FILTER
# ============================================================================
def apply_iqr_outlier_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Interquartile Range (IQR) method to remove outliers for each experience level.
    
    Args:
        df: DataFrame containing salary and experience level data
        
    Returns:
        DataFrame with outliers removed
    """
    print("\n=== APPLYING IQR OUTLIER FILTER ===")
    
    filtered_dfs = []
    total_removed = 0
    
    if 'experience_level' not in df.columns:
        print("Warning: 'experience_level' column not found, skipping IQR filter")
        return df
    
    for exp_level in df['experience_level'].dropna().unique():
        level_data = df[df['experience_level'] == exp_level].copy()
        
        if len(level_data) < 10:  # Skip if too few samples
            filtered_dfs.append(level_data)
            continue
        
        # Calculate IQR
        Q1 = level_data['average_salary'].quantile(0.25)
        Q3 = level_data['average_salary'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (using 1.5 * IQR)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Ensure lower bound is reasonable (not negative)
        lower_bound = max(lower_bound, 1000)
        
        # Filter data
        mask = (level_data['average_salary'] >= lower_bound) & (level_data['average_salary'] <= upper_bound)
        filtered_level = level_data[mask]
        
        removed = len(level_data) - len(filtered_level)
        total_removed += removed
        
        if removed > 0:
            percent_removed = removed / len(level_data) * 100
            print(f"  {exp_level}: Removed {removed:>4,}/{len(level_data):>6,} ({percent_removed:>5.1f}%)")
        
        filtered_dfs.append(filtered_level)
    
    if filtered_dfs:
        result = pd.concat(filtered_dfs, ignore_index=True)
        print(f"\nTotal rows removed by IQR: {total_removed:,}")
        print(f"Final shape after IQR: {result.shape}")
        return result
    else:
        return df


# ============================================================================
# ANALYZE SALARY DISTRIBUTION
# ============================================================================
def analyze_salary_distribution(df: pd.DataFrame, title: str = "Data") -> pd.DataFrame:
    """
    Analyze salary distribution and generate summary statistics.
    
    Args:
        df: DataFrame to analyze
        title: Title for the analysis
        
    Returns:
        Summary statistics dataframe
    """
    print(f"\n=== SALARY ANALYSIS: {title} ===")
    
    if 'experience_level' not in df.columns or 'average_salary' not in df.columns:
        print("Warning: Required columns not found for analysis")
        return pd.DataFrame()
    
    # Generate summary statistics by experience level
    summary = df.groupby('experience_level')['average_salary'].agg([
        ('count', 'size'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75))
    ]).round(2)
    
    # Display summary
    print("\nSummary Statistics by Experience Level:")
    print(summary.to_string())
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total jobs: {len(df):,}")
    print(f"  Overall mean salary: ${df['average_salary'].mean():,.2f}")
    print(f"  Overall median salary: ${df['average_salary'].median():,.2f}")
    print(f"  Overall salary range: ${df['average_salary'].min():,.0f} - ${df['average_salary'].max():,.0f}")
    
    return summary


# ============================================================================
# OPTIMIZE DATAFRAME
# ============================================================================
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize dataframe by downcasting numeric columns and converting objects to categories.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame with smaller memory footprint
    """
    print("\n=== OPTIMIZING DATAFRAME ===")
    
    optimized_df = df.copy()
    initial_memory = optimized_df.memory_usage(deep=True).sum()
    
    # Optimize integer columns
    int_columns = ['minimumYearsExperience', 'numberOfVacancies', 
                   'metadata_totalNumberJobApplication', 'metadata_totalNumberOfView']
    for col in int_columns:
        if col in optimized_df.columns:
            # Downcast to smallest possible integer type
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
    
    # Optimize float columns
    if 'average_salary' in optimized_df.columns:
        optimized_df['average_salary'] = pd.to_numeric(optimized_df['average_salary'], downcast='float')
    
    # Convert object columns with low cardinality to category
    categorical_columns = ['experience_level', 'Industry_Category', 'status_jobStatus']
    for col in categorical_columns:
        if col in optimized_df.columns:
            n_unique = optimized_df[col].nunique()
            if n_unique / len(optimized_df) < 0.5:  # If less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
                print(f"  Converted '{col}' to category: {n_unique} unique values")
    
    # Optimize string columns
    if 'title' in optimized_df.columns:
        optimized_df['title'] = optimized_df['title'].astype('string')
        print("  Converted 'title' to string dtype")
    
    if 'postedCompany_name' in optimized_df.columns:
        optimized_df['postedCompany_name'] = optimized_df['postedCompany_name'].astype('string')
        print("  Converted 'postedCompany_name' to string dtype")
    
    if 'metadata_jobPostId' in optimized_df.columns:
        optimized_df['metadata_jobPostId'] = optimized_df['metadata_jobPostId'].astype('string')
        print("  Converted 'metadata_jobPostId' to string dtype")
    
    # Calculate memory savings
    final_memory = optimized_df.memory_usage(deep=True).sum()
    initial_mb = initial_memory / (1024 * 1024)
    final_mb = final_memory / (1024 * 1024)
    savings = (initial_memory - final_memory) / initial_memory * 100
    
    print(f"\n  Original memory: {initial_mb:.2f} MB")
    print(f"  Optimized memory: {final_mb:.2f} MB")
    print(f"  Memory reduction: {savings:.1f}%")
    
    return optimized_df


# ============================================================================
# SAVE DATASETS
# ============================================================================
def save_datasets(filtered_df: pd.DataFrame) -> None:
    """
    Save the IQR cleaned dataset with only essential columns and optimized data types.
    
    Args:
        filtered_df: DataFrame to save
    """
    filename = "iqr_cleaned.csv"
    print("\n=== Saving iqr_cleaned.csv ===")
    
    # Define columns to keep (essential for analysis)
    columns_to_keep = [
        'metadata_jobPostId',
        'title',
        'postedCompany_name',
        'minimumYearsExperience',
        'experience_level',
        'average_salary',
        'Industry_Category',
        'status_jobStatus',
        'numberOfVacancies',
        'metadata_totalNumberJobApplication',
        'metadata_totalNumberOfView'
    ]
    
    # Keep only columns that exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in filtered_df.columns]
    reduced_df = filtered_df[existing_columns].copy()
    
    print(f"  Selected {len(existing_columns)} columns for output")
    print(f"  DataFrame shape: {reduced_df.shape}")
    print(f"  Columns kept: {existing_columns}")
    
    # Optimize data types for smaller file size
    optimized_df = optimize_dataframe(reduced_df)
    
    # Save to CSV
    optimized_df.to_csv(filename, index=False)
    
    # Calculate file size
    file_size_bytes = os.path.getsize(filename) if os.path.exists(filename) else 0
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"\nSuccessfully saved {filename}")
    print(f"  - Rows: {len(optimized_df):,}")
    print(f"  - Columns: {len(optimized_df.columns)}")
    print(f"  - File size: {file_size_mb:.2f} MB")
    
    
# ============================================================================
# VISUALIZATIONS
# ============================================================================
def create_visualizations(original_df: pd.DataFrame, 
                         filtered_df: pd.DataFrame, 
                         iqr_df: pd.DataFrame) -> None:
    """
    Create visualizations comparing original, filtered, and IQR-cleaned data.
    
    Args:
        original_df: Original dataframe (with outliers)
        filtered_df: Realistically filtered dataframe
        iqr_df: IQR-filtered dataframe
    """
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Check if we have the required columns
    if 'experience_level' not in original_df.columns or 'average_salary' not in original_df.columns:
        print("Warning: Required columns not found for visualizations")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Salary Data Analysis: Before and After Filtering', fontsize=16, fontweight='bold')
    
    datasets = [
        (original_df, 'Original Data (With Outliers)', 'tab:blue'),
        (filtered_df, 'Realistically Filtered', 'tab:green'),
        (iqr_df, 'IQR Filtered (Conservative)', 'tab:orange')
    ]
    
    # Define experience levels order
    exp_levels = [
        '0-5 years', '6-10 years', '11-15 years', '16-20 years',
        '21-25 years', '26-30 years', '31-35 years', '36-40 years', '40+ years'
    ]
    
    # Row 1: Box plots
    for i, (df, title, color) in enumerate(datasets):
        ax = axes[0, i]
        
        # Prepare data for boxplot
        box_data = []
        valid_levels = []
        for level in exp_levels:
            level_data = df[df['experience_level'] == level]['average_salary'].dropna()
            if len(level_data) > 0:
                box_data.append(level_data)
                valid_levels.append(level)
        
        if box_data:
            bp = ax.boxplot(box_data, labels=valid_levels, patch_artist=True)
            # Set box colors
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.6)
            
            ax.set_title(title)
            ax.set_xlabel('Experience Level')
            ax.set_ylabel('Salary (SGD)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add sample size annotation
            total_samples = sum(len(data) for data in box_data)
            ax.text(0.02, 0.98, f'n = {total_samples:,}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Mean salary by experience level
    for i, (df, title, color) in enumerate(datasets):
        ax = axes[1, i]
        
        # Calculate mean salary by experience level
        mean_salaries = df.groupby('experience_level')['average_salary'].mean().reindex(exp_levels)
        
        # Plot
        bars = ax.bar(range(len(exp_levels)), mean_salaries, color=color, alpha=0.7)
        ax.set_title(f'{title}\nMean Salary by Experience')
        ax.set_xlabel('Experience Level')
        ax.set_ylabel('Mean Salary (SGD)')
        ax.set_xticks(range(len(exp_levels)))
        ax.set_xticklabels(exp_levels, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, salary in zip(bars, mean_salaries):
            if not pd.isna(salary):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                       f'${salary:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Row 3: Data retention and additional metrics
    # Subplot 1: Data retention rate
    ax1 = axes[2, 0]
    retention_rates = []
    for level in exp_levels:
        original_count = len(original_df[original_df['experience_level'] == level])
        final_count = len(iqr_df[iqr_df['experience_level'] == level])
        if original_count > 0:
            retention = (final_count / original_count) * 100
        else:
            retention = 0
        retention_rates.append(retention)
    
    bars = ax1.bar(range(len(exp_levels)), retention_rates, color='teal', alpha=0.7)
    ax1.set_title('Data Retention After Filtering')
    ax1.set_xlabel('Experience Level')
    ax1.set_ylabel('Data Retained (%)')
    ax1.set_xticks(range(len(exp_levels)))
    ax1.set_xticklabels(exp_levels, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # Add percentage labels
    for bar, rate in zip(bars, retention_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Salary range by experience (IQR filtered data)
    ax2 = axes[2, 1]
    salary_ranges = []
    for level in exp_levels:
        level_data = iqr_df[iqr_df['experience_level'] == level]['average_salary']
        if len(level_data) > 0:
            salary_range = level_data.max() - level_data.min()
            salary_ranges.append(salary_range)
        else:
            salary_ranges.append(0)
    
    bars = ax2.bar(range(len(exp_levels)), salary_ranges, color='coral', alpha=0.7)
    ax2.set_title('Salary Range by Experience\n(IQR Filtered Data)')
    ax2.set_xlabel('Experience Level')
    ax2.set_ylabel('Salary Range (SGD)')
    ax2.set_xticks(range(len(exp_levels)))
    ax2.set_xticklabels(exp_levels, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, salary_range in zip(bars, salary_ranges):
        if salary_range > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
                   f'${salary_range:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Correlation heatmap (IQR filtered data)
    ax3 = axes[2, 2]
    
    # Prepare data for correlation
    if 'minimumYearsExperience' in iqr_df.columns and 'average_salary' in iqr_df.columns:
        corr_data = iqr_df[['minimumYearsExperience', 'average_salary']].copy()
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax3.set_title('Correlation Heatmap\n(IQR Filtered Data)')
        
        # Set ticks and labels
        tick_labels = ['Years Exp', 'Avg Salary']
        ax3.set_xticks(range(len(tick_labels)))
        ax3.set_yticks(range(len(tick_labels)))
        ax3.set_xticklabels(tick_labels, rotation=45)
        ax3.set_yticklabels(tick_labels)
        
        # Add correlation values
        for i in range(len(tick_labels)):
            for j in range(len(tick_labels)):
                value = corr_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax3.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation Coefficient')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for correlation', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Correlation Heatmap\n(Data Unavailable)')
    
    plt.tight_layout()
    plt.savefig('salary_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: salary_analysis_comparison.png")


def industry_salary_analysis(df: pd.DataFrame) -> None:
    """
    Analyze salary by industry category.
    
    Args:
        df: Cleaned dataframe with industry information
    """
    print("\n=== INDUSTRY SALARY ANALYSIS ===")
    
    # Check if we have the required columns
    if 'Industry_Category' not in df.columns or 'average_salary' not in df.columns:
        print("Warning: Required columns not found for industry analysis")
        return
    
    # Top industries by average salary
    industry_salaries = df.groupby('Industry_Category')['average_salary'].agg([
        ('count', 'size'),
        ('mean_salary', 'mean'),
        ('median_salary', 'median'),
        ('min_salary', 'min'),
        ('max_salary', 'max'),
        ('std_salary', 'std')
    ]).round(2)
    
    # Filter to industries with sufficient data (at least 10 jobs)
    industry_salaries = industry_salaries[industry_salaries['count'] >= 10]
    
    # Sort by mean salary
    industry_salaries = industry_salaries.sort_values('mean_salary', ascending=False)
    
    print("\nTop 10 Industries by Average Salary (min. 10 jobs):")
    print(industry_salaries.head(10).to_string())
    
    print("\nBottom 10 Industries by Average Salary (min. 10 jobs):")
    print(industry_salaries.tail(10).to_string())
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Top 15 industries by average salary
    top_15 = industry_salaries.head(15)
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(top_15)), top_15['mean_salary'], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15.index)
    ax1.set_xlabel('Average Salary (SGD)')
    ax1.set_title('Top 15 Industries by Average Salary\n(Min. 10 job postings)')
    ax1.invert_yaxis()  # Highest at top
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(top_15.iterrows()):
        ax1.text(row['mean_salary'] + 100, i, f'${row["mean_salary"]:,.0f}', 
                va='center', fontsize=9)
        ax1.text(row['mean_salary'] + 100, i + 0.2, f'n={row["count"]:,}', 
                va='center', fontsize=8, alpha=0.7)
    
    # Plot 2: Industry salary by experience level (heatmap)
    ax2 = axes[1]
    
    # Get top 8 industries by count for better readability
    top_8_industries = df['Industry_Category'].value_counts().head(8).index
    
    # Create pivot table
    pivot_data = df[df['Industry_Category'].isin(top_8_industries)]
    
    if 'experience_level' in pivot_data.columns:
        pivot_table = pivot_data.pivot_table(
            values='average_salary',
            index='Industry_Category',
            columns='experience_level',
            aggfunc='mean'
        )
        
        # Reorder columns by experience
        exp_order = [
            '0-5 years', '6-10 years', '11-15 years', '16-20 years',
            '21-25 years', '26-30 years', '31-35 years', '36-40 years', '40+ years'
        ]
        pivot_table = pivot_table.reindex(columns=[col for col in exp_order if col in pivot_table.columns])
        
        # Create heatmap
        if not pivot_table.empty:
            im = ax2.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
            ax2.set_title('Salary Heatmap: Top 8 Industries vs Experience Level')
            ax2.set_xlabel('Experience Level')
            ax2.set_ylabel('Industry')
            
            # Set ticks
            ax2.set_xticks(range(len(pivot_table.columns)))
            ax2.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
            ax2.set_yticks(range(len(pivot_table.index)))
            ax2.set_yticklabels(pivot_table.index)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Average Salary (SGD)')
            
            # Add salary annotations
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.iloc[i, j]
                    if not pd.isna(value):
                        text_color = 'white' if value > pivot_table.values.mean() else 'black'
                        ax2.text(j, i, f'${value:,.0f}', ha='center', va='center', 
                               color=text_color, fontsize=8, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Salary Heatmap\n(Insufficient Data)')
    else:
        ax2.text(0.5, 0.5, 'Experience level data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Salary Heatmap\n(Data Unavailable)')
    
    plt.tight_layout()
    plt.savefig('industry_salary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: industry_salary_analysis.png")
    
    # Print industry insights
    print("\n=== INDUSTRY INSIGHTS ===")
    print(f"Highest paying industry: {industry_salaries.index[0]} (${industry_salaries.iloc[0]['mean_salary']:,.2f})")
    print(f"Lowest paying industry: {industry_salaries.index[-1]} (${industry_salaries.iloc[-1]['mean_salary']:,.2f})")
    print(f"Industry count (min. 10 jobs): {len(industry_salaries)}")
    print(f"Industries with >1000 jobs: {(industry_salaries['count'] > 1000).sum()}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main function to run the complete analysis pipeline."""
    print("=" * 80)
    print("SINGAPORE JOB SALARY ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Database path
    db_path = "/Users/simgsr/Documents/GitHub/NTU_M1_Assignment--main/data/SGJobData_Normalized.db"
    
    try:
        # Step 1: Load data
        jobs_df, categories_df, jobcategories_df = load_data(db_path)
        
        # Step 2: Clean and prepare data - DROP UNUSED COLUMNS EARLY
        cleaned_jobs = clean_and_prepare_data(jobs_df)
        
        # Step 3: Create experience levels
        cleaned_jobs = create_experience_levels(cleaned_jobs)
        
        # Step 4: Create industry categories
        categories_df = create_industry_categories(categories_df)
        
        # Step 5: Join all datasets - only bring in necessary columns
        merged_df = join_datasets(cleaned_jobs, categories_df, jobcategories_df)
        
        # Step 6: Analyze original data (before filtering)
        print("\n" + "=" * 80)
        print("ANALYZING ORIGINAL DATA (BEFORE FILTERING)")
        print("=" * 80)
        original_summary = analyze_salary_distribution(merged_df, "ORIGINAL DATA")
        
        # Step 7: Apply realistic salary filters
        realistically_filtered = apply_realistic_salary_filters(merged_df)
        filtered_summary = analyze_salary_distribution(realistically_filtered, "REALISTICALLY FILTERED")
        
        # Step 8: Apply IQR outlier filter
        iqr_filtered = apply_iqr_outlier_filter(realistically_filtered)
        iqr_summary = analyze_salary_distribution(iqr_filtered, "IQR FILTERED")
        
        # Step 9: Create visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        create_visualizations(merged_df, realistically_filtered, iqr_filtered)
        industry_salary_analysis(iqr_filtered)
        
        # Step 10: Save the IQR cleaned dataset with reduced columns and optimized data types
        save_datasets(iqr_filtered)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  1. iqr_cleaned.csv - Cleaned dataset for modeling")
        print("  2. iqr_cleaned.csv.gz - Compressed cleaned dataset")
        print("  3. salary_analysis_comparison.png - Comparative visualizations")
        print("  4. industry_salary_analysis.png - Industry salary analysis")
            
    except FileNotFoundError:
        print(f"\nError: Database file not found at {db_path}")
        print("Please update the db_path variable with the correct path.")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
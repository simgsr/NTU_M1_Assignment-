import pandas as pd
import numpy as np
import common_utilities as utils

# Suppress scientific notation globally
pd.set_option('display.float_format', '{:.2f}'.format)

# Use read_csv to load the data into a DataFrame
# By default, it assumes the first row is the header
csv_file_path = "data/SGJobData.csv"
df = pd.read_csv(csv_file_path)

#print(df.size)
#print(df.head(5))
#print(df.describe())
#print(df.info(()))
#print(df.columns.tolist())

# check dataframe if columns contain blank entries
utils.check_dataframe_for_blanks(df)

# check categories column for blank entries
utils.check_column_contains_blanks(df['categories'])

# filter dataframe for blank entries in categories column and export to csv
blank_categories_mask = df['categories'].isna() | (df['categories'].astype(str).str.strip() == '')
blank_categories_df = df[blank_categories_mask]
output_file = "data/blank_categories.csv"
#utils.export_dataframe_to_csv(blank_categories_df,output_file)

# filter dataframe for non-blank entries in categories column
filtered_df = df[~blank_categories_mask]
print("filtered df count:\n", filtered_df.count())
print("filtered df size: ", filtered_df.size)

# check filtered dataframe if columns contain blank entries
print("\nAfter removing blank categories:")
utils.check_dataframe_for_blanks(filtered_df)

# export filtered dataframe to csv
output_file = "data/non_blank_categories.csv"
#utils.export_dataframe_to_csv(filtered_df, output_file)

#drop occupationId column as it contains no data
filtered_df = filtered_df.drop(columns=['occupationId'])

#drop status_id column as it contains non-informative data
filtered_df = filtered_df.drop(columns=['status_id'])

# check filtered dataframe if columns contain blank entries after dropping columns
print("\nAfter removing columns occupationId and status_id:")
utils.check_dataframe_for_blanks(filtered_df)

#export filtered dataframe to csv
output_file = "data/filtered_SGJobData.csv"
#utils.export_dataframe_to_csv(filtered_df, output_file)

#check data accuracy for min/max salary
salary_df = filtered_df[['salary_minimum','salary_maximum','average_salary']].astype(np.float64)
invalid_salary_mask = (salary_df['salary_minimum'] > salary_df['salary_maximum'])
invalid_salary_df = salary_df[invalid_salary_mask]
print("\nNumber of entries with invalid salary data:", invalid_salary_df.size, "\nshape:", invalid_salary_df.shape)

#check data accuracy for average salary
invalid_avg_salary_mask = (salary_df['average_salary'] < salary_df['salary_minimum']) | (salary_df['average_salary'] > salary_df['salary_maximum'])
invalid_avg_salary_df = salary_df[invalid_avg_salary_mask]
print("\nNumber of entries with invalid average salary data:", invalid_avg_salary_df.size, "\nshape:", invalid_avg_salary_df.shape)

#check for any duplicate job post id
print("duplicates rows found:",filtered_df.duplicated().sum())
print("duplicates job post ids found:",filtered_df.duplicated(subset=['metadata_jobPostId']).sum())

#check if company name contains similar entries with different cases or symbols
company_names = filtered_df['postedCompany_name'].unique()
company_name_df = pd.DataFrame(company_names, columns=['postedCompany_name'])
company_name_df.sort_values(by=['postedCompany_name'], inplace=True)
#utils.export_dataframe_to_csv(company_name_df,"data/unique_company_names.csv")

# For filtered_df dataframe, convert postedCompany_name column values to uppercase (as strings)
filtered_df['postedCompany_name'] = filtered_df['postedCompany_name'].str.upper()
#export filtered dataframe to csv
output_file = "data/filtered_SGJobData.csv"
#utils.export_dataframe_to_csv(filtered_df, output_file)

#check outliers for salary columns (min, max, average)
print("\nDescribe salary dataframe:\n", salary_df.describe(),"\n", salary_df.info())
print(salary_df.head())

#map employment type to main categories


#map job level to main categories

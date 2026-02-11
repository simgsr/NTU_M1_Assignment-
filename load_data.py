import pandas as pd
import numpy as np
import common_utilities as utils
import re

# Suppress scientific notation globally
pd.set_option('display.float_format', '{:.2f}'.format)

# Use read_csv to load the data into a DataFrame
# By default, it assumes the first row is the header
# remove csv_file_path = "data/SGJobData.csv" to allow user to input their filepath
csv_file_path =  input("Enter the input file path: ").strip()
df = pd.read_csv(csv_file_path)

#print(df.size)
#print(df.head(5))
#print(df.describe())
#print(df.info(()))
#print(df.columns.tolist())

'''
STEP 1 - check dataframe if "categories" column contain blank entries (either empty or string with spaces)
'''
# check dataframe if columns contain blank entries
utils.check_dataframe_for_blanks(df)

# check categories column for blank entries
utils.check_column_contains_blanks(df['categories'])

# filter dataframe for blank entries in categories column and export to csv
blank_categories_mask = (df['categories'].isna() 
                         | (df['categories'].astype(str).str.strip() == ''))
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

'''
STEP 2 - check columns with data that are not meaningful
'''
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

'''
STEP 3 - check data accuracy for min/max/avg salary
'''
#check data accuracy for min/max salary
salary_df = filtered_df[['salary_minimum','salary_maximum','average_salary']].astype(np.float64)
invalid_salary_mask = (salary_df['salary_minimum'] > salary_df['salary_maximum'])
invalid_salary_df = salary_df[invalid_salary_mask]
print("\nNumber of entries with invalid salary data:", invalid_salary_df.size, "\nshape:", invalid_salary_df.shape)

#check data accuracy for average salary
invalid_avg_salary_mask = (salary_df['average_salary'] < salary_df['salary_minimum']) | (salary_df['average_salary'] > salary_df['salary_maximum'])
invalid_avg_salary_df = salary_df[invalid_avg_salary_mask]
print("\nNumber of entries with invalid average salary data:", invalid_avg_salary_df.size, "\nshape:", invalid_avg_salary_df.shape)

'''
STEP 4 - check for any duplicate job post id with "metadata_jobPostId" column
'''
#check for any duplicate job post id
print("duplicates rows found:",filtered_df.duplicated().sum())
print("duplicates job post ids found:",filtered_df.duplicated(subset=['metadata_jobPostId']).sum())

'''
STEP 5 - check if text columns (postedCompany_name, title) contain inconsistent cases and special characters
'''
#check if company name contains similar entries with different cases or symbols
company_names = filtered_df['postedCompany_name'].unique()
company_name_df = pd.DataFrame(company_names, columns=['postedCompany_name'])
company_name_df.sort_values(by=['postedCompany_name'], inplace=True)
#utils.export_dataframe_to_csv(company_name_df,"data/unique_company_names.csv")
company_name_df['postedCompany_name'] = company_name_df['postedCompany_name'].str.upper()
#utils.export_dataframe_to_csv(company_name_df,"data/unique_company_names_upper.csv")

#replace ampersand from company name and title
filtered_df['title'] = filtered_df['title'].str.replace("&", "and")
filtered_df['postedCompany_name'] = filtered_df['postedCompany_name'].str.replace("&", "and")

# Define regex pattern for special characters
# [^a-zA-Z0-9\s] means "any character NOT a letter, digit, whitespace, or period"
special_chars_pattern = r'[^a-zA-Z0-9\s.]'

# Clean the 'Title' and 'Company Name' columns
filtered_df['title'] = filtered_df['title'].str.replace(special_chars_pattern, '', regex=True)
filtered_df['postedCompany_name'] = filtered_df['postedCompany_name'].str.replace(special_chars_pattern, '', regex=True)

# Remove leading and trailing whitespace
filtered_df['title'] = filtered_df['title'].str.strip()
filtered_df['postedCompany_name'] = filtered_df['postedCompany_name'].str.strip()

# Replace multiple spaces with a single space
filtered_df['title'] = filtered_df['title'].str.replace(r'\s+', ' ', regex=True)
filtered_df['postedCompany_name'] = filtered_df['postedCompany_name'].str.replace(r'\s+', ' ', regex=True)

# For filtered_df dataframe, convert postedCompany_name column values to uppercase (as strings)
filtered_df['postedCompany_name'] = filtered_df['postedCompany_name'].str.upper()
#export filtered dataframe to csv
output_file = "data/filtered_SGJobData.csv"
#utils.export_dataframe_to_csv(filtered_df, output_file)

'''
STEP 6 - identify the employment types of jobs to focus on
'''
# group jobs by employment type
employment_type_groupings = filtered_df.groupby('employmentTypes').size().reset_index(name='Count')
employment_type_groupings['Percentage of jobs'] = employment_type_groupings['Count'].astype(np.float64) / len(filtered_df) * 100
employment_type_groupings.sort_values(by='Count', ascending=False, inplace=True)
print("No. of jobs by employment type:\n", employment_type_groupings)

'''
based on employment types, to focus only on 95% of jobs:
- Permanent (44%), Full Time (38%) and Contract (13%)
'''
employment_type_mask = (filtered_df['employmentTypes'].str.fullmatch('Permanent') 
                        | filtered_df['employmentTypes'].str.fullmatch('Full Time') 
                        | filtered_df['employmentTypes'].str.fullmatch('Contract'))
filtered_df = filtered_df[employment_type_mask]

'''
to remove jobs with titles that are meant for excluded employment types e.g. parttime/part time, intern, temp
to remove jobs with titles that contain hourly rates
'''
# regex pattern to identify hourly paid jobs
hourly_paid_jobs_pattern = r'\d+(\.\d+)?\s*(hour|hr|per hour|per hr)'

exclude_job_title_mask = (filtered_df['title'].str.contains('parttime', case=False, na=False) 
                           | filtered_df['title'].str.contains('part time', case=False, na=False) 
                           | filtered_df['title'].str.contains('intern', case=False, na=False) 
                           | filtered_df['title'].str.contains('temp', case=False, na=False)
                           | filtered_df['title'].str.contains('freelance', case=False, na=False)
                           | filtered_df['title'].str.contains('flexible', case=False, na=False)
                           | filtered_df['title'].str.contains(hourly_paid_jobs_pattern, case=False, na=False, regex=True)
                        )

filtered_df = filtered_df[~exclude_job_title_mask]

#to remove outliers where average salary > 400,000
filtered_df = filtered_df[filtered_df['average_salary'] <= 400000]

'''
STEP 7 - identify targeted data set and remove outliers
'''
#to remove outliers where average salary seems too high for job posting
#i.e. metadata_jobPostId = MCF-2024-0351969, MCF-2024-0492535
exclude_specific_jobPostId_mask = (filtered_df['metadata_jobPostId'].str.fullmatch('MCF-2024-0351969')
                                   | filtered_df['metadata_jobPostId'].str.fullmatch('MCF-2024-0492535'))
filtered_df = filtered_df[~exclude_specific_jobPostId_mask]
output_file = "data/filtered_SGJobData.csv"
utils.export_dataframe_to_csv(filtered_df, output_file)

#to remove job postings where years of experience is above 40 years
filtered_df = filtered_df[filtered_df['minimumYearsExperience'] <= 40]
output_file = "data/filtered_SGJobData_40yearsOfExperience.csv"
utils.export_dataframe_to_csv(filtered_df, output_file)

#to remove job postings where average salary is below 3,000
filtered_df = filtered_df[filtered_df['average_salary'] >= 3000]
output_file = "data/filtered_SGJobData_40yearsOfExperience_and_above_3k_salary.csv"
utils.export_dataframe_to_csv(filtered_df, output_file)

'''
Check statistics for salary columns after data cleaning
'''
#check outliers for salary columns (min, max, average)
salary_df = filtered_df[['salary_minimum','salary_maximum','average_salary']].astype(np.float64)
print("\nDescribe salary dataframe:\n", salary_df.describe(),"\n", salary_df.info())
print(salary_df.head())
   

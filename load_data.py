import pandas as pd
import numpy as np
import common_utilities as utils

# Use read_csv to load the data into a DataFrame
# By default, it assumes the first row is the header
csv_file_path = "SGJobData.csv"
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
output_file = "blank_categories.csv"
#utils.export_dataframe_to_csv(blank_categories_df,output_file)

# filter dataframe for non-blank entries in categories column
filtered_df = df[~blank_categories_mask]
print("filtered df count: ", filtered_df.count())
print("filtered df size: ", filtered_df.size)

# check filtered dataframe if columns contain blank entries
print("\nAfter removing blank categories:")
utils.check_dataframe_for_blanks(filtered_df)

# export filtered dataframe to csv
output_file = "non_blank_categories.csv"
#utils.export_dataframe_to_csv(filtered_df, output_file)

#drop occupationId column as it contains no data
filtered_df = filtered_df.drop(columns=['occupationId'])

#drop status_id column as it contains non-informative data
filtered_df = filtered_df.drop(columns=['status_id'])

# check filtered dataframe if columns contain blank entries after dropping columns
print("\nAfter removing columns occupationId and status_id:")
utils.check_dataframe_for_blanks(filtered_df)

#export filtered dataframe to csv
output_file = "filtered_SGJobData.csv"
utils.export_dataframe_to_csv(filtered_df, output_file)
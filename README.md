## SG Job Data Analysis Project

## 📋 Project Overview
This project analyzes Singapore job market data by processing CSV files containing job postings, normalizing the data into a 3NF database schema, and creating comprehensive visualizations for business insights.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- Git (for cloning the repository)

### 2. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate ntu-m1-assignment
```

## Steps performed to clean raw data
### (1) check dataframe if "categories" column contain blank entries (either empty or string with spaces)
- create a boolean masking "blank_categories_mask" to filter out blank entries
- 3,988 rows are blank and are removed from dataset

### (2) check columns with data that are not meaningful
- drop occupationId column as it contains no data
- drop status_id column as it contains non-informative data

### (3) check data accuracy for min/max/avg salary
- salary_minimum cannot be higher than salary_maximum
- average_salary must be between salary_minimum and salary_maximum
- no data issues observed

### (4) check for any duplicate job post id with "metadata_jobPostId" column
- no duplicates observed

### (5) check if text columns (postedCompany_name, title) contain inconsistent cases and special characters
- 447 out of 53,151 company names are in upper case. to standardise all company names to be capitalized
- replace ampersand to "and"
- remove special characters with regex
- remove leading and trailing whitespace
- replace multiple spaces with a single space

### (6) identify the employment types of jobs to focus on
- based on employment types, to focus only on 95% of jobs: Permanent (44%), Full Time (38%) and Contract (13%)
- to remove jobs with titles that are meant for excluded employment types e.g. parttime/part time, intern, temp
- to remove jobs with titles that contain hourly rates
- to remove outliers where average salary > 400,000
- to remove low paying jobs where average salary < 4,000



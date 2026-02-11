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

### 3. Perform EDA
Refer to [EDA.md](EDA.md) for details on cleaning data.

```bash
# run load_data.py to generate cleaned csv file
python load_data.py

# provide the path to the original csv file when asked for input
3 cleaned csv files will be created based on different data set
(a) output_file = "data/filtered_SGJobData.csv" [974,187 records]
(b) output_file = "data/filtered_SGJobData_40yearsOfExperience.csv" [974,173 records]
(c) output_file = "data/filtered_SGJobData_40yearsOfExperience_and_above_3k_salary.csv" [752,316 records]
```

### 4. Create normalized database
```bash
# run m1_gen_db.py to generate the normalized database file "SGJobData_Normalized.db"
python m1_gen_db.py

# provide the path to the cleaned csv file when asked for input
# decide which of the 3 cleaned csv files to be used to create the database 
```

### 5. Connect database with DBGate
Establish a new database connection using "SGJobData_Normalized.db"

Ensure 3 tables are loaded
- Jobs
- Categories
- JobCategories

Run sample sql queries
```sql
select * from categories
select count(*) from jobs
select count(*) from jobcategories
```

### 6. Run Streamlit visualisation
```bash
# run visualisation.py to view the visualisations
streamlit run visualisation.py
```


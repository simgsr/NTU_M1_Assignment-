Prerequisites
Python 3.7+
pip install pandas duckdb numpy

Steps 1
Create data folder & place CSV
In Terminal input - mkdir data
then Copy SGJobData.csv into data/

Steps 2
Activate environment
In Terminal input - conda activate pds

Steps 3
Run script
In Terminal input - python m1_gen_db.py

Steps 4
Enter file path when prompted
In Terminal input - data/SGJobData.csv

The following Output files will be created AND saved in data/ folder
- job_postings.db - DuckDB database
- job_postings_jobs.pkl - Jobs table
- job_postings_categories.pkl - Categories table
- job_postings_job_categories.pkl - Junction table
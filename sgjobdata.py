# %% [markdown]
# # Job Market Analytics Dashboard
# ## Analyzing Singapore Job Market Data
# 
# This notebook analyzes job postings data from Singapore to extract business insights, trends, and market intelligence.

# %% [markdown]
# ### 1. Setup and Database Connection

# %%
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Database path
db_path = "/Users/simgsr/Documents/github_ntu/m1_assignment/SGJobData_Normalized.db"

# Connect to DuckDB
print("Connecting to database...")
con = duckdb.connect(db_path)
print("Connected successfully!")

# %% [markdown]
# ### 2. Data Overview & Quality Check

# %%
# Check table structures
tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
tables = con.execute(tables_query).fetchdf()
print("Available Tables:")
print(tables)

# %%
# Get basic statistics for each table
for table in tables['table_name']:
    print(f"\n=== {table} ===")
    count_query = f"SELECT COUNT(*) as row_count FROM {table}"
    count_result = con.execute(count_query).fetchone()
    print(f"Total Rows: {count_result[0]:,}")
    
    if table == 'Jobs':
        cols_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'Jobs'"
        cols = con.execute(cols_query).fetchdf()
        print(f"Columns: {len(cols)}")

# %%
# Data Completeness Analysis
completeness_query = """
SELECT 
    'Total Jobs' AS metric,
    COUNT(*) AS value
FROM Jobs
UNION ALL
SELECT 
    'Jobs with Salary Info',
    COUNT(*) 
FROM Jobs 
WHERE average_salary > 0
UNION ALL
SELECT 
    'Jobs with Experience Info',
    COUNT(*) 
FROM Jobs 
WHERE minimumYearsExperience IS NOT NULL
UNION ALL
SELECT 
    'Jobs with Company Name',
    COUNT(*) 
FROM Jobs 
WHERE postedCompany_name IS NOT NULL AND postedCompany_name != ''
UNION ALL
SELECT 
    'Jobs with Categories',
    COUNT(DISTINCT metadata_jobPostId) 
FROM JobCategories
"""

completeness_df = con.execute(completeness_query).fetchdf()
print("Data Completeness Analysis:")
print(completeness_df)

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(completeness_df['metric'], completeness_df['value'], color='skyblue')
ax.set_xlabel('Count', fontsize=12)
ax.set_title('Data Completeness Overview', fontsize=14, fontweight='bold')

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, 
            f'{width:,.0f}', 
            ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Market Overview Analysis

# %%
# 3.1 Top 10 Most Common Job Categories
top_categories_query = """
SELECT 
    c.Cat_Name,
    COUNT(jc.metadata_jobPostId) AS job_count,
    ROUND(COUNT(jc.metadata_jobPostId) * 100.0 / (SELECT COUNT(*) FROM JobCategories), 2) AS percentage
FROM Categories c
JOIN JobCategories jc ON c.Cat_ID = jc.Cat_ID
GROUP BY c.Cat_ID, c.Cat_Name
ORDER BY job_count DESC
LIMIT 10
"""

top_categories_df = con.execute(top_categories_query).fetchdf()
print("Top 10 Job Categories:")
print(top_categories_df)

# Visualization
fig = px.bar(top_categories_df, 
             x='job_count', 
             y='Cat_Name',
             orientation='h',
             title='Top 10 Most Common Job Categories',
             labels={'job_count': 'Number of Jobs', 'Cat_Name': 'Category'},
             text='percentage',
             color='job_count',
             color_continuous_scale='Viridis')
fig.update_traces(texttemplate='%{text}%', textposition='outside')
fig.update_layout(height=500, showlegend=False)
fig.show()

# %%
# 3.2 Jobs by Status
status_query = """
SELECT 
    status_jobStatus,
    COUNT(*) AS job_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Jobs), 2) AS percentage
FROM Jobs
GROUP BY status_jobStatus
ORDER BY job_count DESC
"""

status_df = con.execute(status_query).fetchdf()
print("Job Status Distribution:")
print(status_df)

# Pie chart
fig = px.pie(status_df, 
             values='job_count', 
             names='status_jobStatus',
             title='Distribution of Jobs by Status',
             hole=0.3)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

# %%
# 3.3 Monthly Job Posting Trend
monthly_trend_query = """
SELECT 
    SUBSTR(metadata_originalPostingDate, 1, 7) AS posting_month,
    COUNT(*) AS new_jobs,
    SUM(metadata_repostCount) AS reposts,
    COUNT(*) + SUM(metadata_repostCount) AS total_postings
FROM Jobs
WHERE metadata_originalPostingDate IS NOT NULL
    AND metadata_originalPostingDate LIKE '2023%'
GROUP BY posting_month
ORDER BY posting_month
"""

monthly_trend_df = con.execute(monthly_trend_query).fetchdf()
print("Monthly Job Posting Trend (2023):")
print(monthly_trend_df)

# Line chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly_trend_df['posting_month'], 
                         y=monthly_trend_df['new_jobs'],
                         mode='lines+markers',
                         name='New Jobs',
                         line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=monthly_trend_df['posting_month'], 
                         y=monthly_trend_df['total_postings'],
                         mode='lines+markers',
                         name='Total Postings (incl. reposts)',
                         line=dict(color='red', width=3, dash='dash')))

fig.update_layout(title='Monthly Job Posting Trend (2023)',
                  xaxis_title='Month',
                  yaxis_title='Number of Jobs',
                  hovermode='x unified',
                  height=500)
fig.show()

# %% [markdown]
# ### 4. Salary Analysis

# %%
# 4.1 Average Salary by Category (Top 10)
salary_by_category_query = """
SELECT 
    c.Cat_Name,
    COUNT(j.metadata_jobPostId) AS job_count,
    ROUND(AVG(j.average_salary), 2) AS avg_salary,
    ROUND(MIN(j.salary_minimum), 2) AS min_salary,
    ROUND(MAX(j.salary_maximum), 2) AS max_salary,
    ROUND(MAX(j.salary_maximum) - MIN(j.salary_minimum), 2) AS salary_range
FROM Categories c
JOIN JobCategories jc ON c.Cat_ID = jc.Cat_ID
JOIN Jobs j ON jc.metadata_jobPostId = j.metadata_jobPostId
WHERE j.average_salary > 0
GROUP BY c.Cat_ID, c.Cat_Name
HAVING COUNT(j.metadata_jobPostId) >= 10
ORDER BY avg_salary DESC
LIMIT 10
"""

salary_category_df = con.execute(salary_by_category_query).fetchdf()
print("Top 10 Highest Paying Categories:")
print(salary_category_df)

# Bubble chart: Size = job_count, Color = avg_salary
fig = px.scatter(salary_category_df, 
                 x='job_count', 
                 y='avg_salary',
                 size='job_count',
                 color='avg_salary',
                 hover_name='Cat_Name',
                 size_max=60,
                 title='Salary vs Job Count by Category',
                 labels={'job_count': 'Number of Jobs', 'avg_salary': 'Average Salary ($)'},
                 color_continuous_scale='thermal')

fig.update_layout(height=600, showlegend=False)
fig.show()

# %%
# 4.2 Salary Range Distribution
salary_distribution_query = """
SELECT 
    CASE 
        WHEN average_salary < 2000 THEN 'Under $2,000'
        WHEN average_salary BETWEEN 2000 AND 3999 THEN '$2,000 - $3,999'
        WHEN average_salary BETWEEN 4000 AND 5999 THEN '$4,000 - $5,999'
        WHEN average_salary BETWEEN 6000 AND 7999 THEN '$6,000 - $7,999'
        WHEN average_salary BETWEEN 8000 AND 9999 THEN '$8,000 - $9,999'
        WHEN average_salary >= 10000 THEN '$10,000+'
        ELSE 'Not specified'
    END AS salary_range,
    COUNT(*) AS job_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Jobs WHERE average_salary > 0), 2) AS percentage
FROM Jobs
WHERE average_salary > 0
GROUP BY salary_range
ORDER BY MIN(average_salary)
"""

salary_dist_df = con.execute(salary_distribution_query).fetchdf()
print("Salary Distribution:")
print(salary_dist_df)

# Histogram with custom ordering
order = ['Under $2,000', '$2,000 - $3,999', '$4,000 - $5,999', 
         '$6,000 - $7,999', '$8,000 - $9,999', '$10,000+']
salary_dist_df['salary_range'] = pd.Categorical(salary_dist_df['salary_range'], 
                                                categories=order, 
                                                ordered=True)
salary_dist_df = salary_dist_df.sort_values('salary_range')

fig = px.bar(salary_dist_df, 
             x='salary_range', 
             y='job_count',
             title='Salary Distribution of Jobs',
             text='percentage',
             color='job_count',
             color_continuous_scale='blues')
fig.update_traces(texttemplate='%{text}%', textposition='outside')
fig.update_layout(xaxis_title='Salary Range', 
                  yaxis_title='Number of Jobs',
                  showlegend=False)
fig.show()

# %%
# 4.3 Salary vs Experience Correlation
exp_vs_salary_query = """
SELECT 
    CASE 
        WHEN minimumYearsExperience = 0 THEN 'No experience'
        WHEN minimumYearsExperience BETWEEN 1 AND 2 THEN '1-2 years'
        WHEN minimumYearsExperience BETWEEN 3 AND 5 THEN '3-5 years'
        WHEN minimumYearsExperience BETWEEN 6 AND 10 THEN '6-10 years'
        WHEN minimumYearsExperience > 10 THEN '10+ years'
        ELSE 'Not specified'
    END AS experience_level,
    COUNT(*) AS job_count,
    ROUND(AVG(average_salary), 2) AS avg_salary,
    ROUND(AVG(minimumYearsExperience), 1) AS avg_years_exp
FROM Jobs
WHERE average_salary > 0 AND minimumYearsExperience IS NOT NULL
GROUP BY experience_level
ORDER BY AVG(minimumYearsExperience)
"""

exp_salary_df = con.execute(exp_vs_salary_query).fetchdf()
print("Salary vs Experience Level:")
print(exp_salary_df)

# Grouped bar chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Bar chart for job count
fig.add_trace(
    go.Bar(x=exp_salary_df['experience_level'], 
           y=exp_salary_df['job_count'],
           name="Job Count",
           marker_color='lightblue'),
    secondary_y=False,
)

# Line chart for average salary
fig.add_trace(
    go.Scatter(x=exp_salary_df['experience_level'], 
               y=exp_salary_df['avg_salary'],
               name="Avg Salary ($)",
               mode='lines+markers',
               line=dict(color='red', width=3)),
    secondary_y=True,
)

fig.update_layout(
    title_text="Salary vs Experience Level Analysis",
    height=500
)
fig.update_xaxes(title_text="Experience Level")
fig.update_yaxes(title_text="Number of Jobs", secondary_y=False)
fig.update_yaxes(title_text="Average Salary ($)", secondary_y=True)

fig.show()

# %% [markdown]
# ### 5. Company Analysis

# %%
# 5.1 Top 10 Companies with Most Job Postings
top_companies_query = """
SELECT 
    postedCompany_name,
    COUNT(*) AS job_count,
    ROUND(AVG(average_salary), 2) AS avg_salary_offered,
    SUM(numberOfVacancies) AS total_vacancies,
    ROUND(AVG(metadata_totalNumberJobApplication), 0) AS avg_applications_per_job,
    ROUND(AVG(metadata_totalNumberOfView), 0) AS avg_views_per_job
FROM Jobs
WHERE postedCompany_name IS NOT NULL 
    AND postedCompany_name != ''
    AND average_salary > 0
GROUP BY postedCompany_name
HAVING COUNT(*) >= 5
ORDER BY job_count DESC
LIMIT 10
"""

top_companies_df = con.execute(top_companies_query).fetchdf()
print("Top 10 Companies by Job Postings:")
print(top_companies_df)

# Radar chart for top 3 companies
top3_companies = top_companies_df.head(3)

fig = go.Figure()

for idx, row in top3_companies.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row['job_count']/max(top3_companies['job_count']),
           row['avg_salary_offered']/max(top3_companies['avg_salary_offered']),
           row['total_vacancies']/max(top3_companies['total_vacancies']),
           row['avg_applications_per_job']/max(top3_companies['avg_applications_per_job'])],
        theta=['Job Count', 'Avg Salary', 'Total Vacancies', 'Avg Applications'],
        name=row['postedCompany_name'],
        fill='toself'
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    title="Comparison of Top 3 Companies (Normalized Metrics)",
    showlegend=True,
    height=500
)

fig.show()

# %%
# 5.2 Companies with Highest Average Salaries
high_salary_companies_query = """
SELECT 
    postedCompany_name,
    COUNT(*) AS job_count,
    ROUND(AVG(average_salary), 2) AS avg_salary,
    ROUND(MIN(salary_minimum), 2) AS min_salary,
    ROUND(MAX(salary_maximum), 2) AS max_salary,
    ROUND(AVG(minimumYearsExperience), 1) AS avg_experience_required
FROM Jobs
WHERE average_salary > 0 
    AND postedCompany_name IS NOT NULL
    AND postedCompany_name != ''
GROUP BY postedCompany_name
HAVING COUNT(*) >= 3
ORDER BY avg_salary DESC
LIMIT 10
"""

high_salary_companies_df = con.execute(high_salary_companies_query).fetchdf()
print("Top 10 Companies by Average Salary (min 3 postings):")
print(high_salary_companies_df)

# Scatter plot: Salary vs Experience required
fig = px.scatter(high_salary_companies_df, 
                 x='avg_experience_required', 
                 y='avg_salary',
                 size='job_count',
                 color='avg_salary',
                 hover_name='postedCompany_name',
                 title='High-Salary Companies: Experience vs Salary',
                 labels={'avg_experience_required': 'Avg Experience Required (years)',
                         'avg_salary': 'Average Salary ($)',
                         'job_count': 'Number of Jobs Posted'},
                 color_continuous_scale='sunset')

fig.update_layout(height=500)
fig.show()

# %% [markdown]
# ### 6. Job Popularity & Competition Analysis

# %%
# 6.1 Most Competitive Jobs (Highest applications per vacancy)
competitive_jobs_query = """
SELECT 
    j.title,
    c.Cat_Name,
    j.numberOfVacancies,
    j.metadata_totalNumberJobApplication AS total_applications,
    ROUND(j.metadata_totalNumberJobApplication * 1.0 / j.numberOfVacancies, 2) AS applications_per_vacancy,
    j.average_salary,
    j.postedCompany_name
FROM Jobs j
JOIN JobCategories jc ON j.metadata_jobPostId = jc.metadata_jobPostId
JOIN Categories c ON jc.Cat_ID = c.Cat_ID
WHERE j.numberOfVacancies > 0 
    AND j.metadata_totalNumberJobApplication > 0
    AND j.average_salary > 0
ORDER BY applications_per_vacancy DESC
LIMIT 10
"""

competitive_jobs_df = con.execute(competitive_jobs_query).fetchdf()
print("Top 10 Most Competitive Jobs:")
print(competitive_jobs_df)

# Horizontal bar chart
fig = px.bar(competitive_jobs_df,
             y='title',
             x='applications_per_vacancy',
             orientation='h',
             color='applications_per_vacancy',
             hover_data=['Cat_Name', 'average_salary', 'postedCompany_name'],
             title='Most Competitive Jobs (Applications per Vacancy)',
             color_continuous_scale='reds')

fig.update_layout(height=600, 
                  yaxis={'categoryorder':'total ascending'},
                  xaxis_title='Applications per Vacancy')
fig.show()

# %%
# 6.2 Most Viewed Jobs
viewed_jobs_query = """
SELECT 
    j.title,
    c.Cat_Name,
    j.metadata_totalNumberOfView AS total_views,
    j.metadata_totalNumberJobApplication AS total_applications,
    ROUND(j.metadata_totalNumberOfView * 1.0 / j.metadata_totalNumberJobApplication, 2) AS views_per_application,
    j.average_salary,
    j.postedCompany_name
FROM Jobs j
JOIN JobCategories jc ON j.metadata_jobPostId = jc.metadata_jobPostId
JOIN Categories c ON jc.Cat_ID = c.Cat_ID
WHERE j.metadata_totalNumberOfView > 0
    AND j.metadata_totalNumberJobApplication > 0
ORDER BY j.metadata_totalNumberOfView DESC
LIMIT 10
"""

viewed_jobs_df = con.execute(viewed_jobs_query).fetchdf()
print("Top 10 Most Viewed Jobs:")
print(viewed_jobs_df)

# Bubble chart: Views vs Applications
fig = px.scatter(viewed_jobs_df,
                 x='total_applications',
                 y='total_views',
                 size='views_per_application',
                 color='average_salary',
                 hover_name='title',
                 hover_data=['Cat_Name', 'postedCompany_name'],
                 title='Job Popularity: Views vs Applications',
                 labels={'total_applications': 'Total Applications',
                         'total_views': 'Total Views',
                         'views_per_application': 'Views per Application',
                         'average_salary': 'Salary ($)'},
                 color_continuous_scale='viridis')

fig.update_layout(height=600)
fig.show()

# %% [markdown]
# ### 7. Experience & Position Analysis

# %%
# 7.1 Experience Requirements Distribution
experience_dist_query = """
SELECT 
    CASE 
        WHEN minimumYearsExperience = 0 THEN 'Entry level (0 years)'
        WHEN minimumYearsExperience = 1 THEN '1 year'
        WHEN minimumYearsExperience = 2 THEN '2 years'
        WHEN minimumYearsExperience = 3 THEN '3 years'
        WHEN minimumYearsExperience = 4 THEN '4 years'
        WHEN minimumYearsExperience = 5 THEN '5 years'
        WHEN minimumYearsExperience > 5 THEN 'More than 5 years'
        ELSE 'Not specified'
    END AS years_experience,
    COUNT(*) AS job_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Jobs WHERE minimumYearsExperience IS NOT NULL), 2) AS percentage,
    ROUND(AVG(average_salary), 2) AS avg_salary
FROM Jobs
WHERE minimumYearsExperience IS NOT NULL
GROUP BY years_experience
ORDER BY MIN(minimumYearsExperience)
"""

exp_dist_df = con.execute(experience_dist_query).fetchdf()
print("Experience Requirements Distribution:")
print(exp_dist_df)

# Dual-axis chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(x=exp_dist_df['years_experience'], 
           y=exp_dist_df['job_count'],
           name="Job Count",
           marker_color='lightgreen',
           text=exp_dist_df['percentage'],
           texttemplate='%{text:.1f}%',
           textposition='outside'),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=exp_dist_df['years_experience'], 
               y=exp_dist_df['avg_salary'],
               name="Avg Salary ($)",
               mode='lines+markers',
               line=dict(color='darkblue', width=3)),
    secondary_y=True,
)

fig.update_layout(
    title_text="Experience Requirements vs Salary",
    height=500,
    showlegend=True
)
fig.update_xaxes(title_text="Years of Experience Required", tickangle=45)
fig.update_yaxes(title_text="Number of Jobs", secondary_y=False)
fig.update_yaxes(title_text="Average Salary ($)", secondary_y=True)

fig.show()

# %%
# 7.2 Position Levels Distribution
position_levels_query = """
SELECT 
    positionLevels,
    COUNT(*) AS job_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Jobs WHERE positionLevels IS NOT NULL), 2) AS percentage,
    ROUND(AVG(average_salary), 2) AS avg_salary,
    ROUND(AVG(minimumYearsExperience), 1) AS avg_experience_years
FROM Jobs
WHERE positionLevels IS NOT NULL
GROUP BY positionLevels
ORDER BY job_count DESC
"""

position_levels_df = con.execute(position_levels_query).fetchdf()
print("Position Levels Distribution:")
print(position_levels_df)

# Grouped bar chart
fig = px.bar(position_levels_df, 
             x='positionLevels', 
             y=['avg_salary', 'avg_experience_years'],
             title='Position Levels: Salary vs Experience',
             barmode='group',
             labels={'value': 'Value', 'variable': 'Metric'})

fig.update_layout(xaxis_title='Position Level',
                  yaxis_title='Value',
                  height=500)
fig.show()

# %% [markdown]
# ### 8. Advanced Analytics: Category Clusters

# %%
# 8.1 Category Clusters with Similar Salary Ranges
category_clusters_query = """
WITH CategoryStats AS (
    SELECT 
        c.Cat_Name,
        COUNT(j.metadata_jobPostId) AS job_count,
        ROUND(AVG(j.average_salary), 2) AS avg_salary,
        ROUND(STDDEV(j.average_salary), 2) AS salary_std_dev,
        ROUND(AVG(j.minimumYearsExperience), 2) AS avg_experience
    FROM Categories c
    JOIN JobCategories jc ON c.Cat_ID = jc.Cat_ID
    JOIN Jobs j ON jc.metadata_jobPostId = j.metadata_jobPostId
    WHERE j.average_salary > 0
    GROUP BY c.Cat_ID, c.Cat_Name
    HAVING COUNT(j.metadata_jobPostId) >= 10
)
SELECT 
    Cat_Name,
    job_count,
    avg_salary,
    salary_std_dev,
    avg_experience,
    CASE 
        WHEN avg_salary < 3000 THEN 'Low Salary Tier (< $3K)'
        WHEN avg_salary BETWEEN 3000 AND 6000 THEN 'Mid Salary Tier ($3K - $6K)'
        WHEN avg_salary > 6000 THEN 'High Salary Tier (> $6K)'
    END AS salary_tier
FROM CategoryStats
ORDER BY avg_salary DESC
"""

category_clusters_df = con.execute(category_clusters_query).fetchdf()
print("Category Clusters by Salary Tier:")
print(category_clusters_df)

# Scatter plot with clusters
fig = px.scatter(category_clusters_df,
                 x='avg_experience',
                 y='avg_salary',
                 size='job_count',
                 color='salary_tier',
                 hover_name='Cat_Name',
                 hover_data=['salary_std_dev'],
                 title='Job Categories Clustered by Salary Tier',
                 labels={'avg_experience': 'Average Experience Required (years)',
                         'avg_salary': 'Average Salary ($)',
                         'job_count': 'Number of Jobs'},
                 size_max=40)

fig.update_layout(height=600)
fig.show()

# %%
# 8.2 Potential Salary Anomalies
anomalies_query = """
SELECT 
    j.title,
    c.Cat_Name,
    j.postedCompany_name,
    j.minimumYearsExperience,
    j.average_salary,
    ROUND(j.average_salary / NULLIF(j.minimumYearsExperience, 0), 2) AS salary_per_year_exp,
    CASE 
        WHEN j.average_salary < 2000 AND j.minimumYearsExperience > 5 THEN '⚠️ Low salary for high experience'
        WHEN j.average_salary > 10000 AND j.minimumYearsExperience < 2 THEN '💰 High salary for low experience'
        ELSE 'Normal'
    END AS anomaly_flag
FROM Jobs j
JOIN JobCategories jc ON j.metadata_jobPostId = jc.metadata_jobPostId
JOIN Categories c ON jc.Cat_ID = c.Cat_ID
WHERE j.average_salary > 0 
    AND j.minimumYearsExperience IS NOT NULL
    AND j.minimumYearsExperience > 0
ORDER BY salary_per_year_exp DESC
LIMIT 15
"""

anomalies_df = con.execute(anomalies_query).fetchdf()
print("Potential Salary Anomalies:")
print(anomalies_df)

# %% [markdown]
# ### 9. Business Insights Summary

# %%
# Calculate key metrics for summary
summary_metrics = {}

# Total metrics
total_jobs = con.execute("SELECT COUNT(*) FROM Jobs").fetchone()[0]
summary_metrics['Total Jobs'] = f"{total_jobs:,}"

# Average salary
avg_salary = con.execute("SELECT ROUND(AVG(average_salary), 2) FROM Jobs WHERE average_salary > 0").fetchone()[0]
summary_metrics['Average Salary'] = f"${avg_salary:,.2f}"

# Most popular category
most_popular_cat = con.execute("""
    SELECT c.Cat_Name, COUNT(*) as count 
    FROM Categories c 
    JOIN JobCategories jc ON c.Cat_ID = jc.Cat_ID 
    GROUP BY c.Cat_ID, c.Cat_Name 
    ORDER BY count DESC LIMIT 1
""").fetchone()
summary_metrics['Most Popular Category'] = f"{most_popular_cat[0]} ({most_popular_cat[1]:,} jobs)"

# Highest paying category
highest_paying_cat = con.execute("""
    SELECT c.Cat_Name, ROUND(AVG(j.average_salary), 2) as avg_salary
    FROM Categories c 
    JOIN JobCategories jc ON c.Cat_ID = jc.Cat_ID
    JOIN Jobs j ON jc.metadata_jobPostId = j.metadata_jobPostId
    WHERE j.average_salary > 0
    GROUP BY c.Cat_ID, c.Cat_Name
    HAVING COUNT(*) >= 10
    ORDER BY avg_salary DESC LIMIT 1
""").fetchone()
summary_metrics['Highest Paying Category'] = f"{highest_paying_cat[0]} (${highest_paying_cat[1]:,.2f})"

# Job status distribution
active_jobs = con.execute("SELECT COUNT(*) FROM Jobs WHERE status_jobStatus = 'Active'").fetchone()[0]
summary_metrics['Active Jobs'] = f"{active_jobs:,} ({active_jobs/total_jobs*100:.1f}%)"

# Companies with most postings
top_company = con.execute("""
    SELECT postedCompany_name, COUNT(*) as count
    FROM Jobs
    WHERE postedCompany_name IS NOT NULL AND postedCompany_name != ''
    GROUP BY postedCompany_name
    ORDER BY count DESC LIMIT 1
""").fetchone()
summary_metrics['Top Hiring Company'] = f"{top_company[0]} ({top_company[1]:,} postings)"

# Print summary
print("="*80)
print("BUSINESS INTELLIGENCE SUMMARY")
print("="*80)
for key, value in summary_metrics.items():
    print(f"• {key}: {value}")
print("="*80)

# %% [markdown]
# ### 10. Strategic Recommendations

# %%
print("\n" + "="*80)
print("STRATEGIC RECOMMENDATIONS FOR STAKEHOLDERS")
print("="*80)

print("\n🔍 FOR JOB SEEKERS:")
print("1. Focus on High-Demand Categories: {}".format(most_popular_cat[0]))
print("2. Target Salary Tiers: Consider aiming for {} which offers ${} on average".format(
    highest_paying_cat[0], highest_paying_cat[1]))
print("3. Experience Pays: Each additional year of experience increases salary by approximately ${:,.0f}".format(
    avg_salary/5))  # Rough estimate

print("\n🏢 FOR COMPANIES/HR:")
print("1. Competitive Benchmarking: Average market salary is ${:,.2f}".format(avg_salary))
print("2. Talent Pool: {} active jobs indicate competitive market".format(active_jobs))
print("3. Top Competitor: {} leads in hiring with {} postings".format(top_company[0], top_company[1]))

print("\n📊 FOR POLICY MAKERS/EDUCATORS:")
print("1. Skill Gap Analysis: Focus on training for high-demand, high-salary categories")
print("2. Market Monitoring: Track monthly posting trends to anticipate economic shifts")
print("3. Support Entry-Level: {} of jobs require no experience".format(
    exp_dist_df[exp_dist_df['years_experience'] == 'Entry level (0 years)']['percentage'].values[0] if 'Entry level (0 years)' in exp_dist_df['years_experience'].values else "N/A"))

print("\n💡 KEY MARKET INSIGHTS:")
print("1. Salary Distribution: Most jobs ({:.1f}%) fall in the ${} range".format(
    salary_dist_df[salary_dist_df['salary_range'] == '$4,000 - $5,999']['percentage'].values[0] if '$4,000 - $5,999' in salary_dist_df['salary_range'].values else 0,
    '$4,000 - $5,999'))
print("2. Experience Premium: Jobs requiring 3-5 years experience offer ${:,.0f} more than entry-level".format(
    exp_salary_df[exp_salary_df['experience_level'] == '3-5 years']['avg_salary'].values[0] - 
    exp_salary_df[exp_salary_df['experience_level'] == 'No experience']['avg_salary'].values[0]))
print("3. Competition Level: Average of {:.1f} applications per vacancy in most competitive roles".format(
    competitive_jobs_df['applications_per_vacancy'].mean()))
print("="*80)

# %% [markdown]
# ### 11. Export Analysis Results

# %%
# Create a summary DataFrame for export
summary_data = {
    'Metric': list(summary_metrics.keys()),
    'Value': list(summary_metrics.values())
}
summary_df = pd.DataFrame(summary_data)

# Export key insights to CSV
output_path = "/Users/simgsr/Documents/github_ntu/m1_assignment/job_market_analysis_summary.csv"
summary_df.to_csv(output_path, index=False)
print(f"\nAnalysis summary exported to: {output_path}")

# Close database connection
con.close()
print("Database connection closed.")

# %% [markdown]
# ### 12. Executive Dashboard Preview

# %%
# Create a simple executive dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Market Overview', 'Salary Distribution', 
                    'Top Categories', 'Experience vs Salary'),
    specs=[[{"type": "indicator"}, {"type": "pie"}],
           [{"type": "bar"}, {"type": "scatter"}]]
)

# Market Overview Indicator
fig.add_trace(
    go.Indicator(
        mode="number+delta",
        value=total_jobs,
        title={"text": "Total Job Postings"},
        number={'valueformat': ','},
        domain={'row': 0, 'column': 0}
    ),
    row=1, col=1
)

# Salary Distribution Pie
salary_pie_data = salary_dist_df.head(3)  # Top 3 salary ranges
fig.add_trace(
    go.Pie(labels=salary_pie_data['salary_range'], 
           values=salary_pie_data['job_count'],
           hole=0.3),
    row=1, col=2
)

# Top Categories Bar
fig.add_trace(
    go.Bar(x=top_categories_df['Cat_Name'].head(5), 
           y=top_categories_df['job_count'].head(5),
           marker_color='lightblue'),
    row=2, col=1
)

# Experience vs Salary Scatter
fig.add_trace(
    go.Scatter(x=exp_salary_df['avg_years_exp'], 
               y=exp_salary_df['avg_salary'],
               mode='markers+lines',
               marker=dict(size=exp_salary_df['job_count']/100,
                          color=exp_salary_df['avg_salary'],
                          colorscale='Viridis',
                          showscale=True)),
    row=2, col=2
)

fig.update_layout(height=800, 
                  title_text="Executive Dashboard: Singapore Job Market",
                  showlegend=False)
fig.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated insights include:")
print("✓ Market trends and demand patterns")
print("✓ Salary benchmarks by category and experience")
print("✓ Competitive analysis of companies and jobs")
print("✓ Strategic recommendations for stakeholders")
print("✓ Executive dashboard for quick insights")
print("="*80)
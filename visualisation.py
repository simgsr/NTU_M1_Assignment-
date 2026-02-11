import streamlit as st
import pandas as pd
import duckdb

# 1. Establish connection to your database file
conn = duckdb.connect('db/SGJobData_Normalized.db')

# 2. Define your SQL query based on your business objective
categories_sql = """select
                        cat_id as 'Category ID',
                        cat_name as 'Category Name' 
                    from Categories;
                """

sector_jobs_sql = """select c.Sector, count(*) as 'Number Of Job Postings'
                    from JobCategories jc
                    inner join Categories c
                    on jc.Cat_ID = c.Cat_ID
                    group by c.Sector
                    order by count(*) desc;
                """

sector_jobs_status_sql = """
                        select c.Sector, j.status_jobStatus as 'Job Status', count(*) as 'Number Of Job Postings'
                        from JobCategories jc
                        inner join Categories c on jc.Cat_ID = c.Cat_ID
                        inner join Jobs j on jc.metadata_jobPostId = j.metadata_jobPostId
                        group by c.Sector, j.status_jobStatus
                        order by c.Sector;
                    """

# 3. Load the result set directly into a Pandas DataFrame
categories_df = conn.query(categories_sql).to_df()
sector_jobs_df = conn.query(sector_jobs_sql).to_df()
sector_jobs_status_df = conn.query(sector_jobs_status_sql).to_df()

# 4. Streamlit Visualisation
# Sets the page configuration
# You can set the page title and layout here
st.set_page_config(page_title="Singapore Jobs Analytics: Government Insights", layout="wide")
st.title("Singapore Jobs Analytics: Government Insights")
st.caption("NTU SCTP DSAI GROUP 1")
st.header("Dashboard Overview")

st.subheader("All Job Categories")
st.dataframe(categories_df, hide_index=True)

st.subheader("All Job By Sectors")
st.dataframe(sector_jobs_df, hide_index=True)
st.bar_chart(sector_jobs_df, x="Sector", y="Number Of Job Postings", 
             horizontal=True, sort="Number Of Job Postings", width=2000, height="content")

st.subheader("All Job By Sectors and Statuses")
st.dataframe(sector_jobs_status_df, hide_index=True)
st.bar_chart(sector_jobs_status_df, x="Sector", y="Number Of Job Postings", color="Job Status",
             horizontal=True, sort="Number Of Job Postings", width=2000, height="content", stack=False)

# Close connection
conn.close()
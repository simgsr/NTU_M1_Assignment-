import streamlit as st
import pandas as pd
import duckdb

# 1. Establish connection to your database file
conn = duckdb.connect('SGJobData_Normalized.db')

# 2. Define your SQL query based on your business objective
categories_sql = """select
                        cat_id as 'Category ID',
                        cat_name as 'Category Name' 
                    from Categories
                """

# 3. Load the result set directly into a Pandas DataFrame
categories_df = conn.query(categories_sql).to_df()

# 4. Streamlit Visualisation
# Sets the page configuration
# You can set the page title and layout here
st.set_page_config(page_title="Singapore Jobs Analytics: Government Insights", layout="wide")
st.title("Singapore Jobs Analytics: Government Insights")
st.caption("NTU SCTP DSAI GROUP 1")
st.header("Dashboard Overview")
st.subheader("All Job Categories")
st.dataframe(categories_df, hide_index=True)

# Close connection
conn.close()
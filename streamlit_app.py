import streamlit as st
import pandas as pd

# 1. Page Configuration
# This sets the tab title and makes the app use the full width of the screen
st.set_page_config(
    page_title="Data Science Doctor",
    layout="wide"
)

# 2. App Title and Description
st.title("👨‍⚕️ Data Science Doctor: Dataset Viewer")
st.markdown("Upload your CSV file below to inspect the data before processing.")
st.markdown("---") # Adds a horizontal line for separation

# 3. The File Uploader Widget
# We restrict the type to 'csv' so users don't upload images/pdfs by accident
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# 4. Logic to handle the file once uploaded
if uploaded_file is not None:
    try:
        # Read the CSV into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Display Basic Info
        col1, col2 = st.columns(2)
        col1.success(f"File uploaded successfully!")
        col2.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Display the Data
        st.subheader("🔍 Data Preview")
        # st.dataframe is interactive (you can scroll and sort columns)
        st.dataframe(df.head(10), use_container_width=True)

        # Optional: Show column names to help us identify the Target Class later
        with st.expander("See Column Names"):
            st.write(list(df.columns))

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # Message shown when no file is uploaded yet
    st.info("Waiting for file upload...")

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
        # ... (Previous code: st.dataframe(df.head...)) ...

        st.markdown("---")
        st.subheader("📊 Class Imbalance Diagnosis")

        # 1. Select the Target Column
        # We let the user pick which column contains the classes (e.g., 0 vs 1)
        target_column = st.selectbox("Select the Target Column (Class/Label):", df.columns)

        if target_column:
            # 2. Calculate the counts
            class_counts = df[target_column].value_counts()
            
            # Display the raw numbers
            col1, col2 = st.columns(2)
            col1.write("Class Distribution:")
            col1.write(class_counts)

            # 3. Plot the Graph
            # We use Matplotlib explicitly to have control over the chart
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Create a bar plot
            sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis", ax=ax)
            
            ax.set_title(f"Distribution of {target_column}")
            ax.set_ylabel("Number of Examples")
            ax.set_xlabel("Class Label")
            
            # Render the plot in Streamlit
            col2.pyplot(fig)

            # 4. Diagnosis Logic
            # Calculate the ratio to give a "Diagnosis"
            minority_count = class_counts.min()
            majority_count = class_counts.max()
            imbalance_ratio = majority_count / minority_count

            if imbalance_ratio > 1.5:
                st.warning(f"⚠️ Imbalance Detected! The majority class is {imbalance_ratio:.1f}x larger than the minority.")
            else:
                st.success("✅ The dataset looks fairly balanced.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # Message shown when no file is uploaded yet
    st.info("Waiting for file upload...")

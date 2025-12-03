import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Page Config
st.set_page_config(page_title="Data Science Doctor", layout="wide")
st.title("👨‍⚕️ Data Science Doctor: Auto-Encoder & Diagnosis")

# Initialize session state to store processed data for the GAN later
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# 2. File Upload
uploaded_file = st.file_uploader("Upload your Dataset (CSV)", type="csv")

if uploaded_file:
    # Read Data
    df = pd.read_csv(uploaded_file)
    
    # --- AUTO-DETECTION LOGIC ---
    # We automatically grab the last column name as the target
    target_col = df.columns[-1]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info(f"🤖 Auto-Detected Target Column: **'{target_col}'**")
        st.write("Raw Data Preview (First 5 rows):")
        st.write(df.head())

    # --- IMBALANCE DIAGNOSIS ---
    with col2:
        st.subheader("📊 Imbalance Diagnosis")
        
        # Count classes
        counts = df[target_col].value_counts()
        total = len(df)
        
        # Calculate Percentages
        stats_df = pd.DataFrame({
            "Count": counts,
            "Percentage": (counts / total) * 100
        })
        
        # Display Metrics
        st.table(stats_df.style.format({"Percentage": "{:.2f}%"}))
        
        # Visual Check
        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.barplot(x=stats_df.index, y=stats_df['Count'], palette="magma", ax=ax)
        ax.set_title("Class Distribution")
        st.pyplot(fig)

    st.markdown("---")

    # --- AUTO-ENCODING ENGINE ---
    st.subheader("⚙️ Data Processing Engine")
    st.caption("Deep Learning models only understand numbers. Converting text to numbers...")

    # Create a copy so we don't mess up the original display
    df_encoded = df.copy()
    encoders = {} # Dictionary to store our encoders (so we can decode later if needed)

    # Loop through all columns
    for col in df_encoded.columns:
        # Check if the column is Text (Object) or Categorical
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
            le = LabelEncoder()
            # Convert text to numbers
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            # Save the encoder (optional, good practice)
            encoders[col] = le
    
    # Display the result
    st.success("✅ Encoding Complete! Data is ready for the GAN.")
    
    with st.expander("View Encoded Numeric Data (Input for AI)"):
        st.dataframe(df_encoded.head(10), use_container_width=True)
        st.markdown(f"**Shape:** {df_encoded.shape}")

    # Store in session state so the next step (GAN) can access it
    st.session_state.processed_df = df_encoded

    # --- READY FOR NEXT STEP ---
    st.markdown("---")
    st.write("### 🚀 Next Step: Synthetic Data Generation")
    if st.button("Initialize GAN Model"):
        st.info("Pass this `st.session_state.processed_df` to the PyTorch function in the next step.")

else:
    st.info("Please upload a CSV file to begin.")
```

### Key Changes Explained

1.  **`target_col = df.columns[-1]`**:
    This single line looks at the list of column names and grabs the very last one. No user input required.

2.  **The Encoding Loop**:
    ```python
    if df_encoded[col].dtype == 'object':
         df_encoded[col] = le.fit_transform(...)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Page Config
st.set_page_config(page_title="Data Science Doctor", layout="wide")
st.title("👨‍⚕️ This helps to maintain the distribution of the data!!")

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
# ```

# ### Key Changes Explained

# 1.  **`target_col = df.columns[-1]`**:
#     This single line looks at the list of column names and grabs the very last one. No user input required.

# 2.  **The Encoding Loop**:
#     ```python
#     if df_encoded[col].dtype == 'object':
#          df_encoded[col] = le.fit_transform(...)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. THE GENERATOR ---
# Purpose: Takes random noise and tries to turn it into a row of data that looks like the minority class.
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Layer 1: Expand noise to hidden features
            nn.Linear(input_dim, 128),
            nn.ReLU(),      # ReLU activates neurons for non-linearity
            
            # Layer 2: Add complexity
            nn.Linear(128, 256),
            nn.ReLU(),
            
            # Layer 3: Output layer matches your CSV columns
            nn.Linear(256, output_dim),
            nn.Tanh()       # Tanh forces output between -1 and 1 (Required for stability)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. THE DISCRIMINATOR ---
# Purpose: The "Detective". Checks if data is Real (from CSV) or Fake (from Generator).
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Layer 1: Input is a row of data
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2), # LeakyReLU allows small gradients for negative values (prevents dying neurons)
            
            # Layer 2
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: Output is a single number (Probability: Real vs Fake)
            nn.Linear(64, 1),
            nn.Sigmoid()       # Sigmoid squashes output between 0 (Fake) and 1 (Real)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. THE TRAINING LOOP ---
def train_gan_model(data_tensor, epochs=1000, lr=0.001, progress_callback=None):
    """
    Args:
        data_tensor (torch.Tensor): The minority class data (normalized).
        epochs (int): How many times to loop through training.
        lr (float): Learning Rate.
        progress_callback (func): Optional Streamlit function to update progress bar.
    
    Returns:
        generator (model): The trained generator.
        losses (dict): History of Generator and Discriminator error (for plotting).
    """
    
    # Dimensions
    data_dim = data_tensor.shape[1] # Number of columns in CSV
    noise_dim = 10                  # Size of random input vector
    
    # Initialize Networks
    generator = Generator(noise_dim, data_dim)
    discriminator = Discriminator(data_dim)
    
    # Optimizers (Adam is standard for GANs)
    g_optim = optim.Adam(generator.parameters(), lr=lr)
    d_optim = optim.Adam(discriminator.parameters(), lr=lr)
    
    # Loss Function (Binary Cross Entropy)
    loss_fn = nn.BCELoss()
    
    # Trackers for graphing later
    d_losses = []
    g_losses = []
    
    for epoch in range(epochs):
        
        # ==================================================================
        #  TRAIN DISCRIMINATOR (The Detective)
        #  Goal: Maximize probability of correctly classifying Real vs Fake
        # ==================================================================
        
        # 1A. Train on Real Data
        real_data = data_tensor
        real_labels = torch.ones(data_tensor.size(0), 1) # Label = 1 for Real
        
        d_optim.zero_grad()
        output_real = discriminator(real_data)
        d_loss_real = loss_fn(output_real, real_labels)
        
        # 1B. Train on Fake Data
        noise = torch.randn(data_tensor.size(0), noise_dim)
        fake_data = generator(noise)
        fake_labels = torch.zeros(data_tensor.size(0), 1) # Label = 0 for Fake
        
        output_fake = discriminator(fake_data.detach()) # .detach() prevents training G here
        d_loss_fake = loss_fn(output_fake, fake_labels)
        
        # 1C. Update Discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optim.step()

        # ==================================================================
        #  TRAIN GENERATOR (The Counterfeiter)
        #  Goal: Fool the Discriminator (Make it think Fake data is Real)
        # ==================================================================
        
        g_optim.zero_grad()
        
        # We generate new fake data, but this time we want D to output 1 (Real)
        target_labels = torch.ones(data_tensor.size(0), 1) 
        
        # Note: We do NOT detach fake_data here because we want gradients to flow back to G
        output_fake_for_g = discriminator(fake_data) 
        g_loss = loss_fn(output_fake_for_g, target_labels)
        
        g_loss.backward()
        g_optim.step()
        
        # ==================================================================
        #  LOGGING
        # ==================================================================
        if epoch % 50 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            
        # Update Streamlit UI if a callback was provided
        if progress_callback and epoch % 10 == 0:
             progress_callback.progress((epoch + 1) / epochs, text=f"Training... Epoch {epoch}/{epochs}")

    return generator, {"d_loss": d_losses, "g_loss": g_losses}

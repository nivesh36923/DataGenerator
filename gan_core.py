import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. THE GENERATOR ---
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# --- 2. THE DISCRIMINATOR ---
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- 3. TRAINING FUNCTION ---
def train_gan_model(data_tensor, epochs, progress_bar=None):
    """
    Trains the GAN and returns the trained Generator.
    """
    data_dim = data_tensor.shape[1]
    noise_dim = 10
    lr = 0.001
    
    generator = Generator(noise_dim, data_dim)
    discriminator = Discriminator(data_dim)
    
    g_optim = optim.Adam(generator.parameters(), lr=lr)
    d_optim = optim.Adam(discriminator.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        # A. Train Discriminator
        real_data = data_tensor
        fake_noise = torch.randn(data_tensor.size(0), noise_dim)
        fake_data = generator(fake_noise).detach()
        
        d_loss_real = loss_fn(discriminator(real_data), torch.ones(data_tensor.size(0), 1))
        d_loss_fake = loss_fn(discriminator(fake_data), torch.zeros(data_tensor.size(0), 1))
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        # B. Train Generator
        fake_noise = torch.randn(data_tensor.size(0), noise_dim)
        fake_output = generator(fake_noise)
        g_loss = loss_fn(discriminator(fake_output), torch.ones(data_tensor.size(0), 1))
        
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()
        
        # C. Update Progress Bar (if provided)
        if progress_bar and epoch % 10 == 0:
            progress_bar.progress((epoch + 1) / epochs, text=f"Training GAN... Epoch {epoch}/{epochs}")
            
    return generator

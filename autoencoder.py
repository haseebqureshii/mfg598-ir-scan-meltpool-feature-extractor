import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import numpy as np
from PIL import Image

class MeltPoolAutoencoder(nn.Module):
    def __init__(self):
        super(MeltPoolAutoencoder, self).__init__()
        
        # ENCODER: Compresses 224x224 -> Latent Space (512 dims)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512)
        )
        
        # DECODER: Reconstructs Latent Space -> 224x224
        self.decoder_linear = nn.Sequential(
            nn.Linear(512, 256 * 14 * 14),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder_linear(latent)
        x = x.view(-1, 256, 14, 14)
        reconstruction = self.decoder_conv(x)
        return reconstruction

class UnsupervisedH5Dataset(Dataset):
    def __init__(self, h5_path, paths_list, transform=None):
        self.h5_path = h5_path
        self.paths = paths_list
        self.transform = transform
        self.h5f = None 

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.h5f is None: 
            self.h5f = h5py.File(self.h5_path, 'r', swmr=True)
            
        ds = self.h5f[self.paths[idx]]
        random_frame_idx = np.random.randint(0, ds.shape[0])
        
        # 1. Extract raw frame dynamically
        if len(ds.shape) == 3: 
            frame = ds[random_frame_idx, :, :]
        else: 
            frame = ds[random_frame_idx, :, :, 0] # Take first channel
            
        # 2. Defensive Type Casting & Normalization
        # Convert to float32 to avoid int64 (<i8) or 16-bit overflows
        frame = frame.astype(np.float32)
        
        # Normalize frame to 0-255 range (handles both 8-bit and 16-bit sensors)
        f_min, f_max = frame.min(), frame.max()
        if f_max > f_min:
            frame = ((frame - f_min) / (f_max - f_min)) * 255.0
        else:
            frame = np.zeros_like(frame) # Handle completely black/blank frames
            
        # Safely convert to uint8 for PIL
        frame = frame.astype(np.uint8)
        img = Image.fromarray(frame).convert('L')
        
        if self.transform:
            img = self.transform(img)
            
        return img, img

def train_autoencoder(h5_path, paths_list, epochs=10, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = UnsupervisedH5Dataset(h5_path, paths_list, tf)
    # num_workers=0 ensures Sol doesn't crash from too many child processes
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MeltPoolAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(loader)}], MSE Loss: {loss.item():.6f}")

        print(f"==> Epoch {epoch+1} Completed | Average Loss: {running_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "cae_foundation_model.pth")
    print("Training finished. Foundation Model saved.")
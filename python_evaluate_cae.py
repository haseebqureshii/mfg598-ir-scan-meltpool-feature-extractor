import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import getpass
from autoencoder import MeltPoolAutoencoder
from sklearn.decomposition import PCA

def get_sample_frames(h5_path, num_samples=5):
    """Pulls random frames from different sources to test the model."""
    samples = []
    labels = [] # We'll use the 'source' name as a pseudo-label for plotting
    
    with h5py.File(h5_path, 'r', swmr=True) as h5f:
        # Find some valid datasets
        valid_paths = []
        h5f.visititems(lambda n, o: valid_paths.append(n) if isinstance(o, h5py.Dataset) and len(o.shape) >= 3 and o.shape[1] >= 32 else None)
        
        # Prevent ValueError if requested samples exceed available datasets
        actual_samples = min(num_samples, len(valid_paths))
        
        # Pick random datasets
        chosen_paths = np.random.choice(valid_paths, actual_samples, replace=False)
        
        for path in chosen_paths:
            ds = h5f[path]
            frame_idx = np.random.randint(0, ds.shape[0])
            if len(ds.shape) == 3: frame = ds[frame_idx, :, :]
            else: frame = ds[frame_idx, :, :, 0]
            
            # Normalize exactly like we did in training
            frame = frame.astype(np.float32)
            f_min, f_max = frame.min(), frame.max()
            if f_max > f_min: frame = ((frame - f_min) / (f_max - f_min)) * 255.0
            
            img = Image.fromarray(frame.astype(np.uint8)).convert('L')
            samples.append(img)
            labels.append(path.split('/')[0]) # e.g., 'source11'
            
    return samples, labels

def evaluate_and_plot():
    user = getpass.getuser()
    h5_path = f"/scratch/{user}/Final_DatasetV3.h5"
    model_path = "cae_foundation_model.pth"
    
    if not os.path.exists(model_path):
        print("Model weights not found. Run training first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeltPoolAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Generating Reconstructions for the 'Eye Test'...")
    samples, _ = get_sample_frames(h5_path, num_samples=6)
    
    # 1. PLOT RECONSTRUCTIONS
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle("Unsupervised Foundation Model: Melt Pool Reconstructions", fontsize=16)
    
    for i, img in enumerate(samples):
        # Prepare input
        input_tensor = tf(img).unsqueeze(0).to(device)
        
        # Get output
        with torch.no_grad():
            reconstructed_tensor = model(input_tensor)
            
        # Convert back to numpy for plotting
        orig_img = input_tensor.cpu().squeeze().numpy()
        recon_img = reconstructed_tensor.cpu().squeeze().numpy()
        
        axes[0, i].imshow(orig_img, cmap='inferno')
        axes[0, i].set_title("Original Frame")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon_img, cmap='inferno')
        axes[1, i].set_title("CAE Reconstruction")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("reconstruction_results.png", dpi=300)
    print("Saved reconstruction_results.png")

    # 2. PLOT LATENT SPACE (PCA)
    print("Extracting 512-dim Latent Vectors for PCA Clustering...")
    cluster_samples, cluster_labels = get_sample_frames(h5_path, num_samples=200)
    
    latent_vectors = []
    for img in cluster_samples:
        input_tensor = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            # Pass through just the encoder to get the 512 vector
            latent = model.encoder(input_tensor)
            latent_vectors.append(latent.cpu().squeeze().numpy())
            
    latent_vectors = np.array(latent_vectors)
    
    # Reduce 512 dimensions down to 2 for plotting
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(cluster_labels))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, x in enumerate(cluster_labels) if x == label]
        plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], 
                    c=[colors(i)], label=label, alpha=0.7, edgecolors='k')
        
    plt.title("Latent Space Clustering (PCA on 512-Dim Vectors)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("latent_space_pca.png", dpi=300)
    print("Saved latent_space_pca.png")

if __name__ == "__main__":
    evaluate_and_plot()
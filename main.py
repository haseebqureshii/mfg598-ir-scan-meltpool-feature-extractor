import os
import getpass
import h5py
from autoencoder import train_autoencoder

def get_all_dataset_paths(h5_path):
    """
    Crawls the H5 file and strictly filters for valid image datasets.
    It rejects metadata arrays and 1x1 tensors.
    """
    print("Crawling H5 file to map internal data structures...")
    valid_paths = []
    
    with h5py.File(h5_path, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Check 1: Must be at least 3D (Frames, H, W)
                if len(obj.shape) >= 3:
                    # Check 2: Spatial dimensions must be > 32 pixels
                    # obj.shape[1] is Height, obj.shape[2] is Width
                    if obj.shape[1] >= 32 and obj.shape[2] >= 32:
                        valid_paths.append(name)
        f.visititems(visitor)
        
    print(f"Found {len(valid_paths)} strictly valid image sequences.")
    return valid_paths

def main():
    # 1. Path Configuration
    user = getpass.getuser()
    h5_path = f"/scratch/{user}/Final_DatasetV3.h5"

    if not os.path.exists(h5_path):
        print(f"Error: Data not found at {h5_path}.")
        return

    # 2. Extract Valid Image Topology
    h5_internal_paths = get_all_dataset_paths(h5_path)
    
    if not h5_internal_paths:
        print("Error: Could not find valid images. Script aborting.")
        return

    # 3. Start Unsupervised Training
    print("\nStarting Phase 3 Unsupervised Training Workflow...")
    train_autoencoder(h5_path=h5_path, paths_list=h5_internal_paths, epochs=15, batch_size=128)

if __name__ == "__main__":
    main()
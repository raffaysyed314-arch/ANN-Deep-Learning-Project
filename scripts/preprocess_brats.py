import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image

def slice_3d_to_2d():
    print("Starting 3D to 2D slicing...")
    
    # 1. Define where the data is, and where the new 2D images will go
    raw_dir = "data/raw/Task01_BrainTumour/imagesTr"
    out_dir = "data/processed/images"
    os.makedirs(out_dir, exist_ok=True)

    # 2. Find all the 3D brain files
    nifti_files = glob.glob(os.path.join(raw_dir, "*.nii.gz"))
    
    if not nifti_files:
        print("No .nii.gz files found. Ensure the dataset downloaded correctly.")
        return

    print(f"Found {len(nifti_files)} volumes. Slicing the first 50 brains for your dataset...")
    
    # 3. Loop through the files (We use 50 to keep your dataset manageable)
    for file_path in nifti_files[:50]:
        filename = os.path.basename(file_path).split('.')[0]
        
        # Load the 3D volume (The loaf of bread)
        img = nib.load(file_path)
        data_3d = img.get_fdata() 
        
        # BraTS images have 4 dimensions. We grab the first scan type (0) 
        # and slice number 75 (right in the middle of the brain)
        if len(data_3d.shape) == 4:
            slice_2d = data_3d[:, :, 75, 0]
        else:
            slice_2d = data_3d[:, :, 75]
            
        # Normalize the image so the pixels are between 0 and 255
        slice_normalized = ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8) * 255).astype(np.uint8)
        
        # Save the slice as a normal PNG image
        out_file = os.path.join(out_dir, f"{filename}_slice75.png")
        Image.fromarray(slice_normalized).save(out_file)
        
    print(f"Slicing complete. Your 2D images are ready in {out_dir}")

if __name__ == "__main__":
    slice_3d_to_2d()

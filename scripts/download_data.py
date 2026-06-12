import os
import urllib.request
import tarfile

def download_brats():
    print("Initiating direct download of BraTS dataset...")
    target_dir = "data/raw"
    os.makedirs(target_dir, exist_ok=True)
    
    # Direct public AWS S3 link for Medical Segmentation Decathlon (BraTS)
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    file_path = os.path.join(target_dir, "Task01_BrainTumour.tar")
    
    if not os.path.exists(file_path):
        print("Downloading 7GB tar file. This will take a few minutes...")
        # Download without needing any API keys
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print("File already exists.")
        
    print("Extracting files...")
    with tarfile.open(file_path) as tar:
        tar.extractall(path=target_dir)
    print("Extraction complete. BraTS dataset is ready.")

if __name__ == "__main__":
    download_brats()

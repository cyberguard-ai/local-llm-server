import os
from huggingface_hub import hf_hub_download
import shutil

print("Downloading Phi-3 Mini model (MIT License)...")

try:
    downloaded_path = hf_hub_download(
        repo_id='microsoft/Phi-3-mini-4k-instruct-gguf', 
        filename='Phi-3-mini-4k-instruct-q4.gguf',
        cache_dir='./hf_cache'
    )
    print(f"Downloaded to: {downloaded_path}")
    
    # Copy to our models directory
    shutil.copy2(downloaded_path, 'models/model.gguf')
    print("Model copied to models/model.gguf")
    
    # Check file size
    size = os.path.getsize('models/model.gguf')
    print(f"File size: {size:,} bytes ({size / (1024*1024*1024):.2f} GB)")
    
except Exception as e:
    print(f"Error: {e}")
    print("Trying alternative download method...")
    
    # Alternative: Direct download
    import urllib.request
    url = "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, 'models/model.gguf')
    
    size = os.path.getsize('models/model.gguf')
    print(f"Downloaded! File size: {size:,} bytes ({size / (1024*1024*1024):.2f} GB)")
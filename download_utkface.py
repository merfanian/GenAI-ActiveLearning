import requests
import tarfile
import os

# URL of the UTKFace dataset
url = "https://susanqq.github.io/UTKFace/data/UTKFace.tar.gz"

# Path to save the downloaded file
output_filename = "/home/mahdi/Projects/GenAI-ActiveLearning/UTKFace.tar.gz"

# Path to extract the dataset
output_dir = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/utkface"

# Download the file
print(f"Downloading {url}...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_filename, 'wb') as f:
        f.write(response.raw.read())
    print(f"Downloaded {output_filename}")

    # Extract the tar.gz file
    print(f"Extracting {output_filename}...")
    with tarfile.open(output_filename, "r:gz") as tar:
        tar.extractall(path=output_dir)
    print(f"Extracted to {output_dir}")

    # Clean up the downloaded file
    os.remove(output_filename)
else:
    print(f"Failed to download. Status code: {response.status_code}")

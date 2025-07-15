import os
import urllib.request
import tarfile

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"
extract_folder = "aclImdb"

# Download
if not os.path.exists(filename):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filename)

# Extract
if not os.path.exists(extract_folder):
    print("Extracting dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    print("Done.")
else:
    print("Dataset already extracted.")

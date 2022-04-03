import os

import wget
import zipfile

url = 'https://golang.org/dl/go1.17.3.windows-amd64.zip'
download_out_file_path = os.path.join(".", "data", "data.zip")
wget.download(url, out=download_out_file_path)


extract_out_file_path = os.path.join("./", "data")
with zipfile.ZipFile(download_out_file_path) as zip_file:
    zip_file.extractall(extract_out_file_path)
    print("\nExtracted all")

os.remove(download_out_file_path)

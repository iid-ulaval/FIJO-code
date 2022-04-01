import os

import wget
import zipfile

url = 'https://golang.org/dl/go1.17.3.windows-amd64.zip'
download_out_file_path = os.path.join(".", "experiment", "data", "data.zip")
wget.download(url, out=download_out_file_path)


extract_out_file_path = os.path.join("./", "experiment", "data", "fijo.json")
with zipfile.ZipFile(download_out_file_path) as zip_file:
    zip_file.extractall(extract_out_file_path)
    print("\nExtracted all")

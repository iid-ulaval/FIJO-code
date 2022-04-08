import os

import wget
import zipfile

SERVER="dataverse.scholarsportal.info"
PERSISTENT_ID="doi:10.5683/SP3/CHUEJM"
VERSION=1.0

url = f"http://{SERVER}/api/access/dataset/:persistentId/versions/{VERSION}?persistentId={PERSISTENT_ID}"

download_out_file_path = os.path.join(".", "data", "data.zip")
wget.download(url, out=download_out_file_path)


extract_out_file_path = os.path.join("./", "data")
with zipfile.ZipFile(download_out_file_path) as zip_file:
    zip_file.extractall(extract_out_file_path)
    print("\nExtracted all")

os.remove(download_out_file_path)

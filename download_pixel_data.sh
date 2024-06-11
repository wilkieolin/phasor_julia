# Download and extract the smart pixels dataset to the pixel_data directory
# Data repo: https://zenodo.org/records/10783560
mkdir pixel_data
wget -O pixel_data/recon3D.tar.gz https://zenodo.org/records/10783560/files/recon3D.tar.gz?download=1
wget -O pixel_data/labels.tar.gz https://zenodo.org/records/10783560/files/labels.tar.gz?download=1
cd pixel_data
tar -xzvf recon3D.tar.gz
tar -xzvf labels.tar.gz
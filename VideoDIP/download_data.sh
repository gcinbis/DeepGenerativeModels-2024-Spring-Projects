#!/bin/bash

# Function to install gdown if not installed
install_gdown() {
    if ! command -v gdown &> /dev/null
    then
        echo "gdown could not be found, installing..."
        pip install gdown
        if [ $? -ne 0 ]; then
            echo "Failed to install gdown. Please install it manually and re-run the script."
            exit 1
        fi
    else
        echo "gdown is already installed."
    fi
}

# Install gdown
install_gdown

# Download the datasets folder
gdown --folder https://drive.google.com/drive/folders/1aXPEo17npP45v2Fv2TBCRtfdnjb-AqYt?usp=sharing

# Unzip all downloaded zip files in the datasets directory
for file in datasets/*.zip; do
    unzip "$file" -d datasets/
    rm "$file"
done

# Download the tb_logs folder
gdown --folder https://drive.google.com/drive/folders/1dcKk7z7NQI4hveRq_v7KdPnumReiVdVO?usp=sharing

# Unzip all downloaded zip files in the tb_logs directory
for file in tb_logs/*.zip; do
    unzip "$file" -d tb_logs/
    rm "$file"
done

gdown --folder https://drive.google.com/drive/folders/1VuSnPkhwwq3Y-rpPhRxCOe4YX03zt8Cj?usp=sharing
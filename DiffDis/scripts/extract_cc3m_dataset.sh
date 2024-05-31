#!/bin/bash

# Set the directories from the arguments
TRAIN_TAR_DIR=../dataset/cc3m/tar/train
VALIDATION_TAR_DIR=../dataset/cc3m/tar/val
TRAIN_OUTPUT_DIR=../dataset/cc3m/train
VALIDATION_OUTPUT_DIR=../dataset/cc3m/val

# Create output directories if they don't exist
mkdir -p $TRAIN_OUTPUT_DIR
mkdir -p $VALIDATION_OUTPUT_DIR

# Function to extract .tar files from a directory into a specified output directory
extract_tars() {
    local TAR_DIR=$1
    local OUTPUT_DIR=$2
    
    for tar_file in "$TAR_DIR"/*.tar; do
        echo "Extracting $tar_file to $OUTPUT_DIR"
        tar -xvf "$tar_file" -C "$OUTPUT_DIR"
        rm -f "$tar_file"
    done
}

# Extract train .tar files
extract_tars $TRAIN_TAR_DIR $TRAIN_OUTPUT_DIR

# Extract validation .tar files
extract_tars $VALIDATION_TAR_DIR $VALIDATION_OUTPUT_DIR

echo "Extraction complete. Train data is in $TRAIN_OUTPUT_DIR, and validation data is in $VALIDATION_OUTPUT_DIR."

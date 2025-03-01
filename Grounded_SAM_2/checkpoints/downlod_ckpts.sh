#!/bin/bash

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# Download SAM 2.1 checkpoints with --no-check-certificate option
echo "Downloading sam2.1_hiera_tiny.pt checkpoint..."
wget --no-check-certificate $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; exit 1; }

echo "Downloading sam2.1_hiera_small.pt checkpoint..."
wget --no-check-certificate $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url"; exit 1; }

echo "Downloading sam2.1_hiera_base_plus.pt checkpoint..."
wget --no-check-certificate $sam2p1_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2p1_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
wget --no-check-certificate $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "All SAM 2.1 checkpoints are downloaded successfully."


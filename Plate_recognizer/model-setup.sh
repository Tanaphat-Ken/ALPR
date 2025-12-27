#!/bin/bash

# Google Drive file ID
FILE_ID="1yCljBvvD4jPDvGkFcJAsl1Mv9hsy5xDp"

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing..."
    pip install --user gdown
fi

# Download the file using gdown
gdown --id "$FILE_ID"

echo "Download complete."

ZIP_FILE=$(ls | grep '\.zip$')
if [ -n "$ZIP_FILE" ]; then
    echo "Unzipping $ZIP_FILE to models/..."
    mkdir -p models
    unzip "$ZIP_FILE" -d models/
    echo "Unzip complete."
else
    echo "No zip file found."
fi

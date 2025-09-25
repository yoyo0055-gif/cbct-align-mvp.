#!/bin/bash
# Convert CBCT DICOM folder → compressed NIfTI
# Usage: ./convert_dicom.sh <dicom_folder> <output_folder>

dcm2niix -z y -o "$2" "$1"

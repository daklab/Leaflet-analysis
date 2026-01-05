import boto3
import os

# Initialize a boto3 client
s3 = boto3.client('s3')

# Define the S3 bucket name and prefix (adjust this as needed)
bucket_name = 'czb-tabula-muris-senis'
prefix = '10x/30_month/'  # This targets the 10x data for the 1-month folder
main_path = '/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/10x'

# 10X timepoints 
#1_month/ [x]
#18_month/ [x]
#21_month/ [x]
#24_month/ [x]
#3_month/ [x]
#30_month/ [x]

def download_files(bucket, prefix, main_path):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # We are interested in downloading all .bam and .bam.bai files
            if key.endswith('.bam') or key.endswith('.bam.bai'):
                # Construct the local download path by appending the S3 object key to the main_path
                download_path = os.path.join(main_path, key)
                # Check if the file already exists to avoid re-downloading
                if not os.path.exists(download_path):
                    os.makedirs(os.path.dirname(download_path), exist_ok=True)
                    try:
                        print(f"Downloading {key} to {download_path}...")
                        s3.download_file(bucket, key, download_path)
                    except Exception as e:
                        print(f"Failed to download {key}: {str(e)}")  # Print error and skip to next
                else:
                    print(f"File already exists: {download_path}")

print("Starting to download Tabula Muris Senis 10x scRNA-seq data for 1-month group...")
download_files(bucket_name, prefix, main_path)
print("Finished downloading Tabula Muris Senis 10x scRNA-seq data for 1-month group!")

# Instructions to submit this script to the cluster
# Ensure awscli and the correct environment are loaded
# To submit this script to the cluster, use the following command:
# module purge
# module load awscli/1.11.36
# conda activate python3ENV
# sbatch --wrap "python3 /gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/data-installation/002_10X_setup.py" --mem 32G -J AWS_setup

import boto3
import os

# Initialize a boto3 client
s3 = boto3.client('s3')

bucket_name = 'czb-tabula-muris-senis'
prefix = 'Plate_seq/'  # Adjust this prefix as needed
prefix = 'Plate_seq/3_month/'
main_path = '/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS'

## V1: Download all files from the bucket
## def download_files(bucket, prefix, main_path):
##     paginator = s3.get_paginator('list_objects_v2')
##     for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
##         # Filter for .bam and .bam.bai files in results_gencode_ercc directories
##         for obj in page.get('Contents', []):
##             key = obj['Key']
##             if '/results_gencode_ercc/' in key and (key.endswith('.bam') or key.endswith('.bam.bai')):
##                 # Construct the local download path by appending the S3 object key to the main_path
##                 download_path = os.path.join(main_path, key)
##                 # Ensure the directory structure exists
##                 os.makedirs(os.path.dirname(download_path), exist_ok=True)
##                 #print(f"Downloading {key} to {download_path}...")
##                 print("Downloading" + str(key) + "to" + str(download_path) + "...")
##                 s3.download_file(bucket, key, download_path)

## V2: enhance for filtering out already downloaded files and handle exceptions to skip errors without stopping the script
def download_files(bucket, prefix, main_path):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if '/results_gencode_ercc/' in key and (key.endswith('.bam') or key.endswith('.bam.bai')):
                download_path = os.path.join(main_path, key)
                # Check if file already exists to avoid re-downloading
                if not os.path.exists(download_path):
                    os.makedirs(os.path.dirname(download_path), exist_ok=True)
                    try:
                        print(f"Downloading {key} to {download_path}...")
                        s3.download_file(bucket, key, download_path)
                    except Exception as e:
                        print(f"Failed to download {key}: {str(e)}")  # Print error and skip
                else:
                    print(f"File already exists: {download_path}")

print("Let's download Tabula Senis FACS based scRNA-seq data!...")
download_files(bucket_name, prefix, main_path)
print("Finished downloading Tabula Senis FACS based scRNA-seq data!")

# to submit this script to the cluster, use the following command:
# module purge
# module load awscli/1.11.36
# sbatch --wrap "python3 /gpfs/commons/home/kisaev/Leaflet-analysis/TabulaSenis/data-installation/001_AWS_setup.py" --mem 100G -J AWS_setup

## Update: for some reason there is an issue with downloading Month 3 data from AWS.
## need to figure out why at some point for now edited function to download only Month 3 data and skip over files that already exist (from first run) or
## those that cause errors (from second run)
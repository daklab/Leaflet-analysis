module purge
module load awscli/1.11.36

# to check what is there do:
aws s3 ls --no-sign-request s3://czb-tabula-muris-senis/

# the paths we want are:
#1. s3://czb-tabula-muris-senis/Plate_seq/
#2. s3://czb-tabula-muris-senis/Metadata/
#3. s3://czb-tabula-muris-senis/reference-genome/
#4. s3://czb-tabula-muris-senis/Bulk/

# define path where to download the data into
main_path=/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS
cd $main_path

# to download a whole folder do:
aws s3 cp s3://czb-tabula-muris-senis/Metadata/ ./metadata/ --recursive --no-sign-request #[done]
aws s3 cp s3://czb-tabula-muris-senis/reference-genome/ ./reference-genome/ --recursive --no-sign-request 

# ----------------------------------------------------------------------------------
# TO-DO:
# Download bulk at a later point 
# aws s3 cp s3://czb-tabula-muris-senis/Bulk/ ./Bulk/ --recursive --no-sign-request 

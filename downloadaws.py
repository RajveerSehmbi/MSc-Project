import boto3
import botocore
import os

#define the function 
def download_s3_folder(bucket_name, s3_folder, local_folder):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(local_folder, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        bucket.download_file(obj.key, target)
        print(f'Downloaded s3://{bucket_name}/{obj.key} to {target}')

bucket_name = 'msc-data-dae'
s3_folder = 'Download'
local_folder = ''

download_s3_folder(bucket_name, s3_folder, local_folder)

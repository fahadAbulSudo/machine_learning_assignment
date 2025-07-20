import boto3

s3_uri="s3://cv-filtering/input/External Profiles/" # S3 URI of the folder you want to recursively scan, Replace this with your own S3 URI

# Split the s3 uri to extract bucket name and the file prefix
# Splitting S3 URI will generate an array
# Combine the appropirate elements of the array to extraxt BUCKET_NAME and PREFIX
#Around 2+ years of experience in IT Industry, currently looking a position to utilize my
arr=s3_uri.split('/')
bucket =arr[2]
prefix=""
for i in range(3,len(arr)-1):
    prefix=prefix+arr[i]+"/"
    
s3_client = boto3.client("s3",
                                   aws_access_key_id=AWS_ACCESS_KEY,
                                   aws_secret_access_key=AWS_SECRET_KEY,
                                   region_name=AWS_REGION_NAME)

fileDictionary = {}
def list_s3_files_using_client(bucket,prefix,s3_client,fileDictionary):
    response = s3_client.list_objects_v2(Bucket=bucket,  Prefix=prefix) # Featch Meta-data of all the files in the folder
    files = response.get("Contents")
    for file in files: # Iterate through each files
        file_path=file['Key']
        val = file_path.split("/")
        name = val[-1]
        object_url="https://"+bucket+".s3.amazonaws.com/"+file_path
        object_url=object_url.replace(" ","+")  #create Object URL  Manually
        fileDictionary[name] = object_url
        print("Object Url =  "+object_url)
    return fileDictionary

print(len(list_s3_files_using_client(bucket=bucket,prefix=prefix,s3_client=s3_client,fileDictionary=fileDictionary)))
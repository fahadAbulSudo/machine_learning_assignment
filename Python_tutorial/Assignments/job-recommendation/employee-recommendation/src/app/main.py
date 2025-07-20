from fastapi.middleware.cors import CORSMiddleware
from app.routers import skills
from fastapi.params import Body
from fastapi import FastAPI
from mangum import Mangum
import pandas as pd
import uvicorn
import json
import pickle as pkl
import boto3 

id_ = "99999"
secret = "888888"
#token = "token"

'''s3 = boto3.resource('s3',
    aws_access_key_id=id_,
                aws_secret_access_key=secret,
                region_name='ap-south-1')
df = pkl.loads(s3.Bucket("fahad-audit-project").Object("processDF.pkl").get()['Body'].read())'''
'''s3 = boto3.client('s3')
response = s3.get_object(Bucket="cv-filtering", Key='output/AFour Profiles/processDF.pkl')
body = response['Body'].read()
df = pkl.loads(body)'''
data = [10,20,30,40,50,60]
  
# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['Numbers'])


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df1 = df.head(2)

app.include_router(skills.router)

@app.get("/")
def read_root():
    showDFjson = df1.to_json(orient="records")
    recommendations = json.loads(showDFjson)
    return recommendations

#handler = Mangum(app)
'''if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')'''
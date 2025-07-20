from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from fastapi.encoders import jsonable_encoder
from app import schemas
from typing import List, Optional
from botocore.exceptions import ClientError
from loguru import logger
import pandas as pd
#import numpy as np
import pickle as pkl
import json
import os
#from IPython.display import HTML
#import os.path
from app.scoreFunctionList import  recommendTopMatch, similarCandidates
import boto3


#token = "token"

'''bucket_name = 'myawsbucket1098'
file_name = 'sample.txt'

s3_response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
print("s3_response:", s3_response)

file_data = s3_response["Body"].read().decode('utf')
print("file_data:", file_data)'''


'''response = s3client.get_object(Bucket='name_of_your_bucket', Key='path/to_your/file.pkl')

body = response['Body'].read()
data = pickle.loads(body)'''

'''s3 = boto3.resource('s3',
    aws_access_key_id=id_,
                aws_secret_access_key=secret,
                region_name='ap-south-1')'''
'''s3 = boto3.client('s3')
#s3 = boto3.resource('s3')
response = s3.get_object(Bucket="cv-filtering", Key='output/AFour Profiles/processDF.pkl')
body = response['Body'].read()
processDF = pkl.loads(body)'''
#s3 = boto3.resource('s3')
#s31 = boto3.client('s3')



"__________Reading CV___________________"
s3_CV = boto3.client('s3',aws_access_key_id=id_,aws_secret_access_key=secret,region_name='ap-south-1')
response = s3_CV.get_object(Bucket="cv-filtering", Key='output/AFour Profiles/processDF.pkl')
body = response['Body'].read()
processDF = pkl.loads(body)
processDF["id"] = processDF.index

response = s3_CV.get_object(Bucket="cv-filtering", Key='output/External Profiles/processDFExt.pkl')
body = response['Body'].read()
processDFext = pkl.loads(body)
processDFext["id"] = processDFext.index#add id header
#PATH_BASE = os.getcwd()
#print(PATH_BASE)
#processDFext = pkl.load(open('processDFExt1.pkl', 'rb'))

#bucket = s3_CV.Bucket(AWS_Bucket)
#processDF = pkl.loads(s3.Bucket("fahad-audit-project").Object("processDF.pkl").get()['Body'].read())
#PATH_BASE = os.path.dirname(os.path.abspath(__file__))
#PATH_BASE = os.getcwd()
#PATH_PARSE = os.path.join(PATH_BASE,'../output/parseExt.pkl')
#print(PATH_BASE)
#PATH_PROCESSDF = PATH_BASE.replace("\\src","") 
#print(PATH_PROCESSDF)
#PATH_PROCESSDF = PATH_PROCESSDF + '\\profiles\\output\\External Profiles\\processDFExt.pkl'
#PATH_PROCESSDFint = PATH_BASE.replace("\\src","") 
#print(PATH_PROCESSDF)
#PATH_PROCESSDFint = PATH_PROCESSDFint + '\\profiles\\output\\AFour Profiles\\processDF.pkl'
#PATH_PROCESSDF = os.path.join(PATH_BASE,'../../output/processDF.pkl')
#PATH_TEMP = os.path.join(PATH_BASE,'../temp/Rec.pkl')

THRESH_RECOMMENDATION = 70. #0
GREEN_RECOMMEND_MUST = 100. #1
GREEN_RECOMMEND_EXP = 70. #2
YELLOW_RECOMMEND_MUST = 70. #3
shortHeads = ["Name", "Role", "Experience", "File", "Priority", "id"] #4
shortHeadsext = ["Name", "Role", "Experience", "File", "Priority"]
scoreHeads = ["Compatibility %", "Must Have Skills %", "Good to Have %", "Experience Match %"] #5
IDENTIFIER = "id" #6

moreHeads = shortHeads.copy()
moreHeads.append(IDENTIFIER)

#recommendDF = pd.DataFrame()

'''with open(PATH_PROCESSDF, "rb") as file:
  processDFext = pkl.load(file)'''

'''with open(PATH_PROCESSDFint, "rb") as file:
  processDF = pkl.load(file)'''

async def s3_internal_download(suffix:str):
    try:
        key = "input/AFour Profiles/" + suffix
        #key = key.replace(" ", "+")
        print(key)
        return s3_CV.get_object(Bucket=AWS_Bucket, Key=key)['Body'].read() 
    except ClientError as err:
        logger.error(str(err))

async def s3_external_download(suffix:str):
    try:
        key = "input/External Profiles/" + suffix
        #key = key.replace(" ", "+")
        print(key)
        return s3_CV.get_object(Bucket=AWS_Bucket, Key=key)['Body'].read() 
    except ClientError as err:
        logger.error(str(err))

def text_pre_processing(text):
  clean = text.split(",")
  clean = [x.strip() for x in clean]
  return clean

def ID_pre_processing(text):
  clean = text.split(",")
  lst = [int(x) for x in clean]
  return lst

def visualDFMaker(DF1, similar=False):
    DF = DF1.copy(deep=True)
    if not similar:
        passList = DF.index[DF[scoreHeads[0]] >= THRESH_RECOMMENDATION]
        if len(passList) >= 10:
            showDF = DF.loc[passList]#show 10 rows
        else:
            showDF = DF.head(10)
    else:
        showDF = DF
    def f(row):
        if row['Must Have Skills %'] >= 100 and row['Experience Match %'] >= 70:
            val = 'High'
        elif row['Must Have Skills %'] >= 80:
            val = 'Mid'
        else:
            val = 'low'
        return val
    showDF['Priority'] = showDF.apply(f, axis=1)
    showDFShort = showDF[shortHeads]

    return showDFShort, showDF


router = APIRouter(
    prefix="/recommendation/cv",
    tags=['Recommendations']
)

@router.get("/internal", response_model=List[schemas.EmployeeRecommendations])
def get_posts(employeeSkill: schemas.EmployeeSkillBase):
    must_skill = employeeSkill.musthave
    Good2have_skill = employeeSkill.goodhave
    userMust = must_skill
    userGood = Good2have_skill
    minEx = employeeSkill.exp
    recommendDF = recommendTopMatch(userMust, userGood, minEx, processDF)
    showDF, _ = visualDFMaker(recommendDF)
    showDF.loc[showDF["Experience"] == -1, "Experience"] = "Missing Field"
    showDFjson = showDF.to_json(orient="records")
    recommendations = json.loads(showDFjson)
    return recommendations

@router.get("/external", response_model=List[schemas.EmployeeRecommendations])
def get_posts(employeeSkill: schemas.EmployeeSkillBase):
    must_skill = employeeSkill.musthave
    Good2have_skill = employeeSkill.goodhave
    userMust = must_skill
    userGood = Good2have_skill
    minEx = employeeSkill.exp
    #print(processDFext.head())
    recommendDF = recommendTopMatch(userMust, userGood, minEx, processDFext)
    showDF, _ = visualDFMaker(recommendDF)
    showDF.loc[showDF["Experience"] == -1, "Experience"] = "Missing Field"
    showDFjson = showDF.to_json(orient="records")
    recommendations = json.loads(showDFjson)
    return recommendations


@router.get("/internal/similar", response_model=List[schemas.EmployeeRecommendations])
def get_posts(employeeSimilar: schemas.EmployeeSimilar):
    ID = employeeSimilar.id
    must_skill = employeeSimilar.musthave
    Good2have_skill = employeeSimilar.goodhave
    userMust = must_skill 
    userGood = Good2have_skill
    minEx = employeeSimilar.exp
    recommendDF = recommendTopMatch(userMust, userGood, minEx, processDF)
    similarDF = similarCandidates(processDF, recommendDF, ID)
    showDF,_  = visualDFMaker(similarDF, similar=True)
    showDF.loc[showDF["Experience"] == -1, "Experience"] = "Missing Field"
    #showDF['Experience'].mask(showDF['Experience'] == -1, "Missing Field", inplace=True)
    showDFjson = showDF.to_json(orient="records")
    similar = json.loads(showDFjson)
    return similar

@router.get("/external/similar", response_model=List[schemas.EmployeeRecommendations])
def get_posts(employeeSimilar: schemas.EmployeeSimilar):
    ID = employeeSimilar.id
    must_skill = employeeSimilar.musthave
    Good2have_skill = employeeSimilar.goodhave
    userMust = must_skill 
    userGood = Good2have_skill
    minEx = employeeSimilar.exp
    recommendDF = recommendTopMatch(userMust, userGood, minEx, processDFext)
    similarDF = similarCandidates(processDFext, recommendDF, ID)
    showDF,_  = visualDFMaker(similarDF, similar=True)
    showDF.loc[showDF["Experience"] == -1, "Experience"] = "Missing Field"
    showDFjson = showDF.to_json(orient="records")
    similar = json.loads(showDFjson)
    return similar


@router.get("/internal/download/{file_name}")
async def download_internalCV(file_name: str or None = None):
    if not file_name :
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail= "File is not present"
        )
    contents = await s3_internal_download(suffix = file_name)
    return Response(
        content=contents,
        headers={
            'Content-Disposition': f'attachment;filename={file_name}',
            'Content-Type': 'application/octet-stream'
        }
    )

@router.get("/external/download/{file_name}")
async def download_internalCV(file_name: str or None = None):
    if not file_name :
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail= "File is not present"
        )
    contents = await s3_external_download(suffix = file_name)
    return Response(
        content=contents,
        headers={
            'Content-Disposition': f'attachment;filename={file_name}',
            'Content-Type': 'application/octet-stream'
        }
    )
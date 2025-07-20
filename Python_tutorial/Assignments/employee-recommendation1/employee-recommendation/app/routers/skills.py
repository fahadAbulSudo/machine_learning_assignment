from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from fastapi.encoders import jsonable_encoder
from app import schemas
from typing import List, Optional
import pandas as pd
#import numpy as np
import pickle as pkl
import json
import os
#from IPython.display import HTML
#import os.path
from app.scoreFunctionList import  recommendTopMatch, similarCandidates
PATH_BASE = os.path.dirname(os.path.abspath(__file__))
PATH_PROCESSDF = os.path.join(PATH_BASE,'../../output/processDF.pkl')
PATH_TEMP = os.path.join(PATH_BASE,'../temp/Rec.pkl')

THRESH_RECOMMENDATION = 70. #0
GREEN_RECOMMEND_MUST = 100. #1
GREEN_RECOMMEND_EXP = 70. #2
YELLOW_RECOMMEND_MUST = 70. #3
shortHeads = ["Name", "Role", "Experience", "File", "Priority", "id"] #4
scoreHeads = ["Compatibility %", "Must Have Skills %", "Good to Have %", "Experience Match %"] #5
IDENTIFIER = "id" #6

moreHeads = shortHeads.copy()
moreHeads.append(IDENTIFIER)

#recommendDF = pd.DataFrame()

with open(PATH_PROCESSDF, "rb") as file:
  processDF = pkl.load(file)

def text_pre_processing(text):
  clean = text.split(",")
  clean = [x.strip() for x in clean]
  return clean

# user favorite id processing
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
    print(showDF.columns)
    def f(row):
        if row['Must Have Skills %'] >= 100 and row['Experience Match %'] >= 70:
            val = 'High'
        elif row['Must Have Skills %'] >= 80:
            val = 'Mid'
        else:
            val = 'low'
        return val
    showDF['Priority'] = showDF.apply(f, axis=1)
    '''priority = []
    for row in showDF[['Must Have Skills %', 'Experience Match %']]:
        print(row[1])
        if row[0] >= 100 and row[1] >= 70 :    priority.append('High')
        elif row[0] >= 80 and row[0] < 100:   priority.append('Mid')
        else:           priority.append('low')

    showDF['Priority'] = priority'''
    showDFShort = showDF[shortHeads]

    return showDFShort, showDF

router = APIRouter(
    prefix="/recommendation",
    tags=['Recommendations']
)

@router.get("/internalCV", response_model=List[schemas.EmployeeRecommendations])
def get_posts(employeeSkill: schemas.EmployeeSkillBase):
    must_skill = employeeSkill.musthave
    Good2have_skill = employeeSkill.goodhave
    userMust = must_skill
    userGood = Good2have_skill
    minEx = employeeSkill.exp
    #global recommendDF
    recommendDF = recommendTopMatch(userMust, userGood, minEx, processDF)
    #pkl.dump(recommendDF, open(PATH_TEMP, 'wb'))
    showDF, _ = visualDFMaker(recommendDF)
    showDFjson = showDF.to_json(orient="records")
    recommendations = json.loads(showDFjson)
    return recommendations


@router.get("/similar", response_model=List[schemas.EmployeeRecommendations])
def get_posts(employeeSimilar: schemas.EmployeeSimilar):
    ID = employeeSimilar.id
    must_skill = employeeSimilar.musthave
    #question_name = post.question_name
    Good2have_skill = employeeSimilar.goodhave
    userMust = must_skill
    userGood = Good2have_skill
    minEx = employeeSimilar.exp
    recommendDF = recommendTopMatch(userMust, userGood, minEx, processDF)
    #print(recommendDF)
    #global recommendDF
    #df1 = recommendDF.merge(df)
    #ID = df1['original id'].tolist()
    similarDF = similarCandidates(processDF, recommendDF, ID)
    showDF,_  = visualDFMaker(similarDF, similar=True)
    showDFjson = showDF.to_json(orient="records")
    similar = json.loads(showDFjson)
    return similar
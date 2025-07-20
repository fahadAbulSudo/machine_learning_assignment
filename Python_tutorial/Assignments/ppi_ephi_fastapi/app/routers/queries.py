from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from fastapi.responses import JSONResponse
from typing import Dict, Any
from app.sql_connecter import get_schema_info
from app import schemas, oauth2
from typing import List, Optional
import pandas as pd
import json
import os

router = APIRouter(
    prefix="/database",
    tags=['DatabaseQuery']
)

@router.post("/query")
def get_posts(queryInfo: schemas.Qyery_Input, current_user: dict = Depends(oauth2.get_current_user)):
    with open("app/data.json", "r") as json_file:
        data = json.load(json_file)
    database = data['database']
    username = data['username']
    password = data['password']
    uri = data['uri']
    query = queryInfo.Query
    bool, df = get_schema_info(uri, username, password, database, query)
    dbjson = df.to_json(orient="records")
    response = json.loads(dbjson)
    if bool is False:
        response_data = {
            "error": "Data not found for query",
            "database info": response
        }

        raise HTTPException(status_code=404, detail=response_data)
    else:  
        return response

@router.post("/databaseInfo", response_model=schemas.DatabaseInfoResponse)
def get_posts(databaseInfo: schemas.DatabaseInfo, current_user: dict = Depends(oauth2.get_current_user)):
    database = databaseInfo.DatabaseName
    username = databaseInfo.DatabaseUsername
    password = databaseInfo.DatabasePassword
    uri = databaseInfo.DatabaseHost
    data = {
        "database": database,
        "username": username,
        "password": password,
        "uri": uri
    }
    databaseInfo = {
        "DatabaseName": database,
        "DatabaseUsername": username,
    }
    try:
        with open("app/data.json", 'w') as json_file:
            json.dump(data, json_file)
    except Exception as e:
        print("Error:", str(e))
    return databaseInfo

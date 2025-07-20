#from distutils.util import execute
from fastapi import Depends, FastAPI, status, HTTPException, APIRouter
from schemas import UserSchema, UserSchemaIn
from typing import List
from db import User, database
from passlib.hash import pbkdf2_sha256

router = APIRouter(tags=["Users"])

@router.post('/users/', status_code=status.HTTP_201_CREATED, response_model=UserSchemaIn)
async def add_user(user:UserSchema):
    hashed_password = pbkdf2_sha256.hash(user.password)
    query = User.insert().values(username=user.username, password=hashed_password)
    last_record_id = await database.execute(query)
    return {**user.dict(), "id": last_record_id}

@router.get('/users/', response_model=List[UserSchemaIn])
async def get_userss():
    query = User.select()
    return await database.fetch_all(query=query) 
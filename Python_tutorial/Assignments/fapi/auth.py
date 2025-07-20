from distutils.util import execute
from fastapi import Depends, FastAPI, status, HTTPException, APIRouter
from schemas import LoginSchema
from typing import List
from db import User, database
from passlib.hash import pbkdf2_sha256
from Token import create_access_token
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(tags=["Auth"])

@router.post('/login/')
async def login(request:OAuth2PasswordRequestForm = Depends()):
    query = User.select().where(User.c.username == request.username)
    myuser = await database.fetch_one(query=query)

    if not myuser:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The user does not exist")
    if not pbkdf2_sha256.verify(request.password, myuser.password):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid Password")

    access_token = create_access_token(
        data={"sub": myuser.username})
    return {"access_token": access_token, "token_type": "bearer"}
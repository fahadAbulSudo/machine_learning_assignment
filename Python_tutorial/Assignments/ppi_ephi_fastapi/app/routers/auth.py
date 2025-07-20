from fastapi import APIRouter, Depends, status, HTTPException, Response
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from app import schemas, utils, oauth2
import json

USER_FILE = "app/users.json"

router = APIRouter(tags=['Authentication'])

def read_users():
    with open(USER_FILE, "r") as file:
        users = json.load(file)
    return users

@router.post('/login', response_model=schemas.Token)
def login(user_credentials:  OAuth2PasswordRequestForm = Depends()):
    users = read_users()
    user = None
    for stored_user in users:
        if stored_user["email"] == user_credentials.username:
            user = stored_user
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials")

    if not utils.verify(user_credentials.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials")

    access_token = oauth2.create_access_token(data={"username": user.username}) 
    return {"access_token": access_token, "token_type": "bearer"}
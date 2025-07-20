from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter
from app import schemas, utils
import json
import re

USER_FILE = "app/users.json"

router = APIRouter(
    prefix="/users",
    tags=['Users']
)

def read_users():
    with open(USER_FILE, "r") as file:
        users = json.load(file)
    return users

def write_users(users):
    with open(USER_FILE, "w") as file:
        json.dump(users, file, indent=2)

#This is for creating user ID
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def create_user(user: schemas.UserCreate):

    users = read_users()
    for existing_user in users:
        if existing_user["username"] == user["username"]:
            raise HTTPException(status_code=400, detail="Username already taken")

    password_pattern = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$")
    if not password_pattern.match(user["password"]):
        raise HTTPException(status_code=400, detail="Password does not meet complexity requirements")

    # hash the password - user.password
    hashed_password = utils.hash(user.password)
    user.password = hashed_password
    users.append(user)
    write_users(users)
    newuser = {
        "username": user.username
    }
    return newuser

@router.patch('/{username}')
def get_user(user: schemas.UserCreate):
    password_pattern = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$")
    if not password_pattern.match(user["password"]):
        raise HTTPException(status_code=400, detail="Password does not meet complexity requirements")
    new_password = user.password
    hashed_password = utils.hash(new_password)
    new_password = hashed_password
    users = read_users()
    user_found = False
    for update_user in users:
        if update_user["username"] == user.username:
            update_user["password"] = new_password
            user_found = True
            break
    if not user_found:
        raise HTTPException(status_code=404, detail="User not found")
    
    write_users(users)

    return {"message": f"Password updated successfully for user {user.username}"}
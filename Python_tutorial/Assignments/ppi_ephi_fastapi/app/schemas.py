from pydantic import BaseModel, EmailStr
from datetime  import datetime
from typing import Union, List, Optional

class DatabaseInfo(BaseModel):
    DatabaseName: str
    DatabaseUsername: str
    DatabasePassword: str
    DatabaseHost: str

class DatabaseInfoResponse(BaseModel):
    DatabaseName: str
    DatabaseUsername: str
    class Config:
        orm_mode = True
    
class Qyery_Input(BaseModel):
    Query: str 

class UserOut(BaseModel):
    username: EmailStr
    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    username: EmailStr
    password: str

class UserLogin(BaseModel):
    username: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
"""
This file contains all the schemas or we can say Pydantic models
"""
from pydantic import BaseModel
from typing import List, Optional


class BlogBase(BaseModel):
    """
    Pydantic Schema for blog
    """
    title: str
    body: str


class Blog(BlogBase):
    """
    Blog class for relationship
    """
    class Config:
        """
        Config class to define the orm mode
        """
        orm_mode = True


class User(BaseModel):
    """
    Pydantic Schema for User
    """
    name: str
    email: str
    password: str


class ShowUser(BaseModel):
    """
    This is a class to show specific information of an User.
    Used as response model.
    """
    name: str
    email: str
    blogs: List[Blog] = []

    class Config:
        """
        Config class to define the orm mode
        """
        orm_mode = True


class ShowBlog(BaseModel):
    """
    This is a class to show specific information of a blog.
    Used as response model.
    """
    title: str
    body: str
    creator: ShowUser

    class Config:
        """
        Config class to define the orm mode
        """
        orm_mode = True


class Login(BaseModel):
    """
    Login class for authentication
    """
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None

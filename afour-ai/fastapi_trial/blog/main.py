"""
This is the main file to perform crud operations on blog
"""
from fastapi import FastAPI
from . import models
from .database import engine
from .routers import authentication, blog, user


app = FastAPI()

# This line actually creates the tables
models.Base.metadata.create_all(bind=engine)

app.include_router(blog.router)
app.include_router(user.router)
app.include_router(authentication.router)

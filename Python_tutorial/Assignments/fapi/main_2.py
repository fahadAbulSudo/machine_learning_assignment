from distutils.util import execute
from fastapi import Depends, FastAPI, status, HTTPException
import db
import articles, users, auth, image_upload

db.metadata.create_all(db.engine)
app = FastAPI()

@app.on_event("startup")
async def startup():
    await db.database.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.database.disconnect()

app.include_router(articles.router)
app.include_router(users.router)
app.include_router(auth.router)
app.include_router(image_upload.router)

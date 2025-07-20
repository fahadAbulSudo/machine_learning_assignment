from fastapi.middleware.cors import CORSMiddleware
from app.routers import queries, user, auth
from fastapi.params import Body
from fastapi import FastAPI
import uvicorn
import json



app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(queries.router)
app.include_router(user.router)
app.include_router(auth.router)

@app.get("/")
def read_root():
    return {"Hello": "world"}

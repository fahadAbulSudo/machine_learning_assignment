from fastapi.middleware.cors import CORSMiddleware
from app.routers import skills
from fastapi.params import Body
from fastapi import FastAPI
#from mangum import Mangum



app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(skills.router)

@app.get("/")
def read_root():
    return {"Hello": "world"}

#handler = Mangum(app)



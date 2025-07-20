from fastapi import FastAPI
from pydantic import BaseModel

class Article(BaseModel):
    id:int
    title:str
    description:str

app = FastAPI()
data = [{"Book":"HP"},{"Course":"Python"}]

@app.get("/")
async def index():
    return {"Message":"welcome"}

@app.get('/article/{id}')
def get_article(id:int):
    return {"article":{id}}

@app.get('/article/')
def get_article(skip:int = 0, limit:int = 20):
    return data[skip : skip + limit]

@app.post('/article/')
def add_article(article:Article):
    return article
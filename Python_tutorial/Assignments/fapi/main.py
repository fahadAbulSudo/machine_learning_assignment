from http.client import HTTPException
from turtle import title
from fastapi import Depends, FastAPI, status, HTTPException
from database import engine, sessionLocal
import models
from schemas import ArticleSchema, MyArticleSchema
from sqlalchemy.orm import Session
from typing import List

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
@app.get("/")
async def index():
    return {"Message":"Good_Morning"}

def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get('/articles/', response_model=List[MyArticleSchema])#in response model we do not want the primary key 
def get_article(db:Session = Depends(get_db)):   # means "id" as output
    myarticles = db.query(models.Article).all()
    return myarticles

#This code is for getting only one reading at a time
@app.get('/articles/{id}', status_code=status.HTTP_200_OK, response_model=MyArticleSchema)#in response model here we do not added List because we want only one article not all bunch 
def article_details(id:int, db:Session = Depends(get_db)):
    #myarticle = db.query(models.Article).filter(models.Article.id == id).first()
    myarticle = db.query(models.Article).get(id) #this is simple than above method
    if myarticle:
        return myarticle

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The article does not exist")

@app.post('/article/', status_code=status.HTTP_201_CREATED)
def add_article(article:ArticleSchema, db:Session = Depends(get_db)):
    new_article = models.Article(title=article.title)
    db.add(new_article)
    db.commit()
    db.refresh(new_article)
    return new_article

@app.put('/article/{id}', status_code=status.HTTP_202_ACCEPTED)
def update_article(id:int, article:ArticleSchema, db:Session = Depends(get_db)):
    db.query(models.Article).filter(models.Article.id == id).update({'title':article.title})
    return {'message':'the data is updated'}

@app.delete('/article/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_article(id:int, article:ArticleSchema, db:Session = Depends(get_db)):
    db.query(models.Article).filter(models.Article.id == id).delete(synchronize_session=False)
    db.commit()
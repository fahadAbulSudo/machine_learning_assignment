from distutils.util import execute
from fastapi import Depends, FastAPI, status, HTTPException, APIRouter
from schemas import ArticleSchema, ArticleSchemaIn, UserSchemaIn
from typing import List
from db import Article, database
from Token import get_current_user


router = APIRouter(tags=["Articles"])

@router.post('/article/', status_code=status.HTTP_201_CREATED, response_model=ArticleSchemaIn)
async def add_article(article:ArticleSchema):
    query = Article.insert().values(title=article.title)
    last_record_id = await database.execute(query)
    return {**article.dict(), "id": last_record_id}

@router.get('/articles/', response_model=List[ArticleSchemaIn])
async def get_articles(current_user:UserSchemaIn = Depends(get_current_user)):
    query = Article.select()
    return await database.fetch_all(query=query) 

@router.get('/articles/{id}', response_model=ArticleSchemaIn)
async def get_details(id:int):
    query = Article.select().where(id == Article.c.id)
    myarticle = await database.fetch_one(query=query)

    if not myarticle:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The article does not exist")

    return {**myarticle}

@router.put('/articles/{id}', response_model=ArticleSchemaIn)
async def update_article(id:int, article:ArticleSchema):
    query = Article.update().where(Article.c.id == id).values(title=article.title)
    await database.execute(query)
    return {**article.dict(), "id": id}

@router.delete('/articles/{id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_article(id:int):
    query = Article.delete().where(Article.c.id == id)
    await database.execute(query)
    return {"message":"Article deleted"}

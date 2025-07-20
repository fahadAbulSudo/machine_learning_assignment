from pydantic import BaseModel

class ArticleSchema(BaseModel):
    title:str

class ArticleSchemaIn(ArticleSchema): #Response model 
    id:int

class UserSchema(BaseModel):
    username:str
    password:str

class UserSchemaIn(BaseModel): #Response model 
    id:int
    username:str
    
class LoginSchema(BaseModel):
    username:str
    password:str

class TokenData(BaseModel):
    username: str | None = None


class MyArticleSchema(ArticleSchema):  #in response model we do not want the primary key
    title:str                        # means "id" as output

    class Config:
        orm_mode=True

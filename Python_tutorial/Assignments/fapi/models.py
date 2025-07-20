from database import Base
from sqlalchemy import Column, Integer, String, Table, MetaData

class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100))
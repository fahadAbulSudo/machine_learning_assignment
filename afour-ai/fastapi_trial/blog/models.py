"""
This file contains all the models in the database
"""
from sqlalchemy import Column, ForeignKey, Integer, String
from .database import Base
from sqlalchemy.orm import relationship


class Blog(Base):
    """
    This is the blog class
    """
    __tablename__ = 'blogs'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    body = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))

    creator = relationship("User", back_populates="blogs")


class User(Base):
    """
    This is the user class
    """
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    password = Column(String)

    blogs = relationship("Blog", back_populates="creator")


"""
Router for blogs
"""
from fastapi import APIRouter, Depends, status
from typing import List
from ..schemas import Blog, ShowBlog, User
from ..database import get_db
from sqlalchemy.orm import Session
from ..repository import blog
from ..oauth2 import get_current_user

router = APIRouter(
    prefix="/blog",
    tags=['Blogs']
)


@router.get('/', response_model=List[ShowBlog])
def get_all_blogs(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Function to fire a query to get all blogs from the database
    :param db: database instance
    :return: all blogs
    """
    return blog.get_all(db)


@router.post('/', status_code=status.HTTP_201_CREATED)
def create(request: Blog, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Function to create a post request to add a new blog
    :return: created blog
    """
    return blog.create(request, db)


@router.get('/{id}', status_code=status.HTTP_200_OK, response_model=ShowBlog)
def get_blog_by_id(id, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Function to fire a query to get a blog by id from the database
    :param id: request id
    :param db: database instance
    :return: blog
    """
    return blog.get_by_id(id, db)


@router.delete('/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_blog(id, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Function to fire a query for delete the blog from the database
    :param id: request id
    :param db: db instance
    :return: deleted status
    """
    return blog.delete(id, db)


@router.put('/{id}', status_code=status.HTTP_202_ACCEPTED)
def update_blog(id, request: Blog, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Function to fire a query for update the blog from the database
    :param id: blog id
    :param request: request
    :param db: db instance
    :return: update status
    """
    return blog.update(id, request, db)


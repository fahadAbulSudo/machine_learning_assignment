"""
Repository file for blogs
"""
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from .. import models, schemas


def get_all(db: Session):
    """
    Get all blogs and return
    :param db: db instance
    :return: all blogs
    """
    blogs = db.query(models.Blog).all()
    return blogs


def get_by_id(id, db: Session):
    """
    Get a blog by its id and return
    :param id: blog id
    :param db: db instance
    :return: blog
    """
    blog = db.query(models.Blog).filter(models.Blog.id == id).first()
    if not blog:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Blog with id {id} is not available')
    return blog


def create(request: schemas.Blog, db: Session):
    """
    Create blog and return
    :return: Created blog
    """
    new_blog = models.Blog(title=request.title, body=request.body, user_id=1)
    db.add(new_blog)
    db.commit()
    db.refresh(new_blog)
    return new_blog


def delete(id, db: Session):
    """
    Delete a particular blog
    :param id: blog id
    :param db: db instance
    :return:
    """
    blog = db.query(models.Blog).filter(models.Blog.id == id)
    # Check if blog with the id is available, if not, raise the exception
    if not blog.first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Blog with id {id} is not available')
    blog.delete(synchronize_session=False)
    db.commit()
    return f'Blog with id {id} deleted'


def update(id: int, request: schemas.Blog, db: Session):
    """
    Update the information of a particular blog
    :param request: request body
    :param id: blog id
    :param db: db instance
    :return: Updated status
    """
    blog = db.query(models.Blog).filter(models.Blog.id == id)
    # Check if blog with the id is available, if not, raise the exception
    if not blog.first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Blog with id {id} is not available')
    blog.update(request.__dict__)
    db.commit()
    return "Updated"


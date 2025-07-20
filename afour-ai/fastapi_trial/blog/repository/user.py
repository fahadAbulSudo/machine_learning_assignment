"""
Repository file for Users
"""
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from .. import models
from ..schemas import User
from ..hashing import Hash


def create(request: User, db: Session):
    """
    Create a new user
    :return: new user
    """
    new_user = models.User(name=request.name, email=request.email, password=Hash.bcrypt(request.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


def get_all(db: Session):
    """
    Get all users from the database
    :return: all users
    """
    users = db.query(models.User).all()
    return users


def get_by_id(id: int, db: Session):
    """
    Get a user by its id
    :param id: user id
    :param db: db instance
    :return: user
    """
    user = db.query(models.User).filter(models.User.id == id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'User with id {id} is not available')
    return user

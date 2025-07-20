"""
Router for users
"""
from fastapi import APIRouter, Depends, status
from typing import List
from ..schemas import User, ShowUser
from ..database import get_db
from sqlalchemy.orm import Session
from ..repository import user

router = APIRouter(
    prefix="/user",
    tags=["Users"]
)


@router.post('/', status_code=status.HTTP_201_CREATED)
def create(request: User, db: Session = Depends(get_db)):
    """
    Function to create a post request to add a new user
    :return: created user
    """
    return user.create(request, db)


@router.get('/', response_model=List[ShowUser])
def get_all_users(db: Session = Depends(get_db)):
    """
    Function to fire a query to get all users from the database
    :param db: database instance
    :return: all users
    """
    return user.get_all(db)


@router.get('/{id}', status_code=status.HTTP_200_OK, response_model=ShowUser)
def get_user_by_id(id, db: Session = Depends(get_db)):
    """
    Function to fire a query to get a user by id from the database
    :param id: request id
    :param db: database instance
    :return: blog
    """
    return user.get_by_id(id, db)

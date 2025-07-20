from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# in "pwd_context" is telling that we want to hash our password in the form of CryptContext("bcrypt")


def hash(password: str):
    return pwd_context.hash(password)
    #Here it is for only to hashing the password while creating user.

def verify(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)
    #This code is for checking the password while login so it hashed the password here and verify with 
    #the password in the database
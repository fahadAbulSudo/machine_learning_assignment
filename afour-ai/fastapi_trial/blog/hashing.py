"""
This file contains a class called Hash which is used to encrypt the password
"""
from passlib.context import CryptContext

pwd_cxt = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Hash:
    """
    Class hash
    """
    def bcrypt(password: str):
        """
        Function to hash the password
        :return:
        """
        return pwd_cxt.hash(password)

    def verify(hashed_password, plain_password):
        """

        :param plain_password:
        :return:
        """
        return pwd_cxt.verify(plain_password, hashed_password)

from pydantic import BaseModel
from datetime  import datetime
from typing import Union, List


class EmployeeSkillBase(BaseModel):
    musthave: List[str]
    goodhave: List[str]
    exp: float

class EmployeeSimilar(BaseModel):
    musthave: List[str]
    goodhave: List[str]
    exp: float
    id: List[int]
    #Name: str
    #Role: str
    #Experience: float

#class Employee(BaseModel):
    #data: List[EmployeeBase]



class EmployeeRecommendations(BaseModel):
    Name: str
    Role: Union[str, None]
    Experience: Union[float, str]
    File: str
    Priority: str 
    id: int
    class Config:
        orm_mode = True


#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_3

# In[11]:


#1.	Write a program to illustrate the use of standard libraries 
#(list, tuple, range, set, dict, str, int, math, random, enum, os, sys)


# In[20]:


import math
import random
import enum
import os
import sys
#List libraries
A = [1,3,23,323]
A.append(67)
print(A.pop())
#Tuple library
B= (342,43,5,3,43,43)
print(B.count(43))
print(B.index(43))
#Range libraries
Range = range(1,8)
print(list(Range))
#Set libraries
S = set()
S.add(1)
S.add(4)
S.add(8)
print(S)
M = set()
M.add(1)
M.add(10)
M.add(820) 
print(S.union(M))
print(S.intersection(M))
#Dict libraries
Dic = {"Name":"Sam", "Subject":"Mathematics","Marks":78}
print(Dic.items())
print(Dic.keys())
Dic.update({"Rank":4})
print(Dic)
#Str libraries
str = "rjgjgrjgkjkfjg"
print(str.count("f"))
print(str.capitalize())
print(str.swapcase())
#int libraries
INT = 567
print(bin(INT))
#Math libraries
Mat = 42
print(pow(Mat,2))
print(pow(Mat,3))
#Random libraries
randNo = random.randint(1, 452)
print(randNo)
#Enum libraries
class flower(enum.Enum):
    lily= 1
    rose= 2
    tulip= 3
 
print ("The string representation of enum member is : ",end="")
print (flower.lily)
print ("The repr representation of enum member is : ",end="")
print (repr(flower.lily))
#OS libraries
import os
root = r'C:\Users\abulf\OneDrive\Desktop'
for root,dirs,files in os.walk('Test', topdown=True):
    print ('--------------------------------')
    for name in files:
           print(os.path.join(root, name))
    for name in dirs:
           print(os.path.join(root, name))
    print ('--------------------------------')
#Sys libraries


print(sys.version)  
  
for line in sys.stdin: 
    if 'q' == line.rstrip(): 
        break
    print(f'Input : {line}') 
  
print("Exit") 


# In[1]:


#5.	Write a function to compute 5/0 and use try/except to catch the exceptions?


# In[4]:


def devision():
    try:
        print(5/0)
    except ZeroDivisionError:
        print("Please do not divide by zero")

devision()


# In[5]:


#6.	Write a program to raise a custom exception which takes a string message as attribute?


# In[13]:


class CustomException(Exception):

    def __init__(self, num, message):
        self.num = num
        self.message = message
        super(CustomException, self).__init__(self.message)

    def __str__(self):
        return f"{self.num} is not prime number. {self.message}"


num = int(input("enter a prime number : "))
flag = False

if num > 1:
    for k in range(2, num):
        if (num % k) == 0:
            flag = True
            break
if num == 1 or flag is True: 
    raise CustomException(num, "please enter prime number")


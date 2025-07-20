#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_1

# In[1]:


#5.	Write a program which accepts a sequence of comma-separated numbers from console and generate a list, tuple 
#and a set which contains every number as per their properties?


# In[2]:


sample_set = {10, 20, 30, 40}
l = []
for i in sample_set:
    l.append(i)
print(l)


# In[3]:


values = input("Input some comma seprated numbers : ")
list = values.split(",")
tuple = tuple(list)
print('List : ',list)
print('Tuple : ',tuple)


# In[5]:


#6.	Create an iterator that returns numbers, starting with 1, and each sequence will increase by one (returning 1,2,3,4,5...) 
#and raise StopIteration exception when the number is greater than 10 ?


# In[12]:


class number:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        if self.a <= 10:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration

myclass = number()
myiter = iter(myclass)

for x in myiter:
    print(x, end=' ')


# In[13]:


#7.	Program to find the sum of first 100 natural numbers using range() ?


# In[18]:


sum = 0
for j in range(1,101):
    sum = sum + j
    if j<=99:
        print(sum, end=',')
    else:
        print(sum)


# In[19]:


#8.	Program to create a copy of an object1 to object2 and append new element to object2


# In[1]:


import numpy as np
import copy
R = int(input("Enter the number of rows:"))
matrix = []
for k in range(0,R):          
    sab = k+1
    a =[]
    for m in range(3):     
         a.append(sab)
    matrix.append(a)
for k in range(R):
    for m in range(3):
        print(matrix[k][m], end = " ")
    print()
matrix_1 = copy.deepcopy(matrix)
print(k)
k+=2 
matrix = np.array(matrix)
r = np.array([k,k,k])
matrix_1 = np.append(matrix_1,[r],axis= 0)
print(matrix)
print(matrix_1)


# In[ ]:





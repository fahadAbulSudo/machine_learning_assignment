#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_8

# In[1]:


#1.	Program to make use of lambda with filter to get the numbers divisible by 11 from the list
#Input -  [1,3,4,5,6,7,8,9,22,25,33,34]
#Output - [22,33


# In[7]:


list1 = []
num = int(input("Enter the number of elements to be added in list: "))
for i in range(0,num):
    val = int(input())
    list1.append(val)
list2 = list(filter(lambda x : x % 11 == 0, list1))
print(list2)


# In[8]:


#2.	Program to make use of lambda with map to get the square of each element of the list
#Input -  [1,3,4,5,6]
#Output - [1,9,16,25,36]


# In[15]:


list1 = []
num = int(input("Enter the number of elements to be added in list: "))
for i in range(0,num):
    val = int(input())
    list1.append(val)
list2 = list(map(lambda x : x **2, list1)) #we do not print direct map func bcoz it return address and type
print(list2)


# In[16]:


#3.	Program to make use of lambda with map to replace characters of string i.e. replace ‘u’ with ‘o’
#Input - Somesh and Poshkar
#Output - Sumesh and Pushkar


# In[21]:


str1 = input("Enter a sring: ")
a1 = input("Character which has to be replaced: ")
a2 = input("Character which has to be replaced with: ")
str2 = ''.join(map(lambda x: x if(x != a1 and x != a2) else a1 if x == a2 else a2, str1))
print(str2)


# In[22]:


#4.	Program to make use of lambda with reduce to get the sum of all elements of the list
#Input -  [1,3,4,5,6]
#Output - 19


# In[24]:


from functools import reduce
list1 = []
num = int(input("Enter the number of elements to be added in list: "))
for i in range(0,num):
    val = int(input())
    list1.append(val)
Sum = reduce( (lambda x, y: x + y), list1)
print(Sum)


# In[25]:


#5.	Write a program which can map() and filter() to make a list whose elements are square 
#of even numbers in [1,2,3,4,5,6,7,8,9,10].


# In[28]:


from math import sqrt
list1 = []
num = int(input("Enter the number of elements to be added in list: "))
for i in range(0,num):
    val = int(input())
    list1.append(val)
list2 = list(filter(lambda x : x % 2 == 0, list1))
list2 = list(map(lambda x : x **2, list2))
print(list2)


# In[32]:


from math import sqrt
list1 = []
num = int(input("Enter the number of elements to be added in list: "))
for i in range(0,num):
    val = int(input())
    list1.append(val)
list2 = list(filter(lambda number : int(sqrt(number) + 0.5) ** 2 == number, list1))
list2 = list(filter(lambda x : x % 2 == 0, list2))
print(list2)


# In[33]:


#6.	Program to zip and unzip the values -
#Input:
#variable = ['x', 'y', 'z']
#value = [3, 4, 5, 0, 9]

#Output: - 
#[('x', 3), ('y', 4), ('z', 5)]
#variable = ('x', 'y', 'z')
#value = (3, 4, 5)


# In[36]:


variable = ['x', 'y', 'z']
value = [3, 4, 5, 0, 9]

zipped = zip(variable, value)
zipped_list = list(zipped)
print (zipped_list)

variable, value = zip(*zipped_list)
print(variable)
print(value)


# In[37]:


#7.	Create 2 directories and add few sample files in those directories and 
#write a setup.py file to package everything as a whl file


# In[ ]:





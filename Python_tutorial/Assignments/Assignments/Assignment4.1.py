#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_4

# In[1]:


#1.	Write one liner list comprehension that takes a list 
#and makes a new list that contains the square of all the even elements of this list?


# In[3]:


k = int(input("Enter number of elements : "))
c = []

for l in range(0, k):
    el = int(input())
    c.append(el)
d = [(val, pow(val,2)) for val in c]
print(d)


# In[4]:


#2.	Write a program to create a list of tuples from given list having number 
#and its square in each tuple? Also, convert the list of tuples to a dictionary. (Using Comprehension)


# In[7]:


k = int(input("Enter number of elements : "))
c = []

for l in range(0, k):
    el = int(input())
    c.append(el)
d = [(val, pow(val,2)) for val in c]
print(d)
dict = {}
{dict.setdefault(a, []).append(b) for a, b in d}
print (dict)


# In[8]:


#3.	Create a Set comprehension to get the unique words in the string.


# In[11]:


s = input()
s = s.lower()
setting = {x for x in s.split(" ")}
print(setting)


# In[12]:


#4.	Create a List comprehension to get the files with .txt extension.


# In[26]:


import os
file_path = input()
list1 = [file for file in os.listdir(file_path) if file.endswith(".txt")]
print(list1)


# In[27]:


#5.	Create a generator expression to find the sum of cube of first 20 elements?


# In[29]:


def cubic_generator(n):
    count = 0
    sum = 0
    for i in range(20):
        yield i ** 3
        count +=1

sum = 0
num = 0
for i in cubic_generator(20):
    sum = sum + i
    num +=1
    if num == 20:
        print(sum)
    else:
        print(sum,  end=' : ')  
        


# In[30]:


#6.	Create a Dictionary comprehension to get the length of each word in the list?


# In[32]:


s = input()
s = s.split(" ")
Dict = {word : len(word) for word in s}
print(Dict)


# In[33]:


#7.	Write a List comprehension to get numbers that are multiples of 2 as well as 3 upto 50.


# In[34]:


list1 = [x for x in range (51) if x%2 == 0 or x%3 == 0]
print(list1)


# In[ ]:





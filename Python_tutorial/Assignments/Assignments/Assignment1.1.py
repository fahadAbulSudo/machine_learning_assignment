#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_1

# In[58]:


#1.	Write any python program that makes use of variables, constants, operators, atleast 5 keywords and print
#statements of different forms ?
# for constant theer is no keyword in python but to set a constant value we should use a "getter_function" to read the data bt not "setter_function" to write the data


# In[35]:


from math import sqrt
a = []
b = []
count = 0
for i in range(10,1001):
    a.append(i)
length= len(a)
for i in range(length):
    flag = True
    m = a[i]
    for j in range(2, m):
        if (a[i] %j== 0):
            if a[i] == 45:
                print(a[i])
                print(j)
                print(a[i] %j)
            flag = False
            break
    if flag:
        count+=1
        b.append(a[i])
print(count)
print(b)
        
    


# In[50]:


#2.	Write a program that takes a list of integers and gives the number of occurrences of each element in the list ?


# In[45]:


#1st Method

def counter(e, x):
    count = 0
    for ele in e:
        if (ele == x):
            count = count + 1
    return count
 
e = []
  
n = int(input("Enter number of elements : "))
  

for o in range(0, n):
    ele = int(input())
    a.append(ele) 
y = all([isinstance(item, int)for item in a])
if y is True:  
    print(a)
    x = int(input("Enter nymber which has to be counted"))
    print(f"{x} has occurred {counter(a, x)} times")
else :
    print("Please enter integers in the list")


# In[44]:


##2nd method

def counter(a, x):
    return a.count(x)
 
a = []
  
n = int(input("Enter number of elements : "))
  

for i in range(0, n):
    ele = int(input())
    a.append(ele) 
y = all([isinstance(item, int)for item in a])
if y is True:  
    print(a)
    x = int(input("Enter nymber which has to be counted"))
    print(f"{x} has occurred {counter(a, x)} times")
else :
    print("Please enter integers in the list")


# In[51]:


#3.	Write a one liner program to reverse the string?


# In[49]:


str = input()
print(str[::-1])


# In[52]:


#4.	Write a program to create a list of tuples from given list having number and its square in each tuple?


# In[57]:


k = int(input("Enter number of elements : "))
c = []

for l in range(0, n):
    el = int(input())
    c.append(el)
d = [(val, pow(val,2)) for val in c]
print(d)


# In[ ]:





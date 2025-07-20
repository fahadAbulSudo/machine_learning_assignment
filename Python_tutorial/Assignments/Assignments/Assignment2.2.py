#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_2

# In[1]:


#6.	Write a program to find the maximum of 2 numbers using normal function 
#and then using ternary operator and then illustrate the same using lamdba function ?


# In[4]:


#ternary
a = int(input()) 
b = int(input())
  

maximum = a if a > b else b
  
print(maximum)


# In[5]:


#normal function
def maximum(a,b):
    if a>b:
        return a
    else:
        return b
a = int(input()) 
b = int(input())
print(maximum(a,b))


# In[7]:


#lamda function
a = int(input()) 
b = int(input())
maximum = lambda a,b :  a if a > b else b #for lamda we normaly use ternary function
print(maximum(a,b))


# In[ ]:


#7.	Write a program using loops and closure to find the multipliers of 4 (4,8,12,16,....,40)?


# In[10]:


def make_multiplier_of(n):   #in closure function, function returns the function name which has been nested in it 
    def multiplier(x):
        return x * n
    return multiplier


times4 = make_multiplier_of(4)# Multiplier of 4
for ao in range(1,11):
    if (ao < 10):
        print(times4(ao), end = ",")
    else:
        print(times4(ao), end = " ")


# In[11]:


#8.	Write a program to illustrate the use of default, keyword, optional and variable length args (*args and **kwargs)?


# In[14]:


# keyword type 
def f(a, b, c):
    print(a+ b+ c)
f(c=3, b=2, a=1)


# In[16]:


# default type and optional type is default type
def f(a, b = 5, c = 8):
    print(a+ b+ c)
f(1)
f(1,3)


# In[17]:


#variable length type
def fnc(*args, **kwargs):
    print('{} {}'.format(args, kwargs))

print('fnc()')
fnc()
fnc(1,2,3)
fnc(1,2,3,'flask')
fnc(a=1, b=2, c=3)
fnc(a=1, b=2, c=3, d='ansible')
fnc(1, 2, 3, a=1, b=2, c=3)


# In[18]:


#9.	Write a function that will take one list as input and return three different types of list 
#as illustrated in the output. In object2, you can append a list containing triplet of a number 
#but object 3 should be evaluated based on the elements present in the object2 
#(Note:You have to take care of both the space and time complexities as well)
#Input:
#object1 = [[1,1,1],[2,2,2],[3,3,3]]

#Output:
#object1 = [[1,1,1],[2,2,2],[3,3,3]]
#object2 = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
#object3 = [[1,1,1],[2,4,2],[3,9,3],[4,16,4]]


# In[21]:


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
        print(matrix[k][m], end = ",")
    print(",")
matrix_1 = copy.deepcopy(matrix)
print(k)
k+=2 
matrix = np.array(matrix)
r = np.array([k,k,k])
R+=1
matrix_1 = np.append(matrix_1,[r],axis= 0)
matrix_2 = copy.deepcopy(matrix_1)
for k in range(0,R):          
    sab = k+1
    #a =[]
    for m in range(3):     
        if m == 1: 
            matrix_2[k][m] = sab **2
    #else
    #matrix_2.append(a)
print(matrix)
print(matrix_1)
print(matrix_2)


# In[23]:


#10.	Write a program to find the sum of the digits of the number recursively upto n iterations?
       #Input 
#	Enter the number -139
#Enter iterations - 2
 
#Output:
#           13
#	4


# In[13]:


def reverse(ab,bc):
    bc-=1
    sum = 0
    while (ab!=0):
        sum = sum + (ab%10)
        ab = int(ab/10)
    ab = sum
    m.append(sum)
    if bc > 0:
        reverse(ab,bc)
    else:
        return

m = []
ab = int(input("Enter the number: "))
bc = int(input("Enter iterations: "))
cd = bc
reverse(ab,bc)
for aoc in range(0,cd):
    print(f"Sum of iteration number {aoc+1} is {m[aoc]}") 


# In[ ]:





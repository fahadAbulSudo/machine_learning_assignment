#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_6

# In[1]:


#1.	Write a recursive function to find the factorial of the specified number. 
#Also, create a decorator that will validate the following before finding the factorial:
#a.	Type of the argument
#b.	No should not be negative 
#c.	Other corner case


# In[9]:


import functools
def Negative_int(func):
    @functools.wraps(func)
    def wrapper(arg):

        try:
            num = int(arg)
            if num <= 0:
                raise ValueError("That is not a positive number!")
        except ValueError as ve:
            return print(ve)
        else:
            return func(num)
    return wrapper

def type_of_argument(func):
    @functools.wraps(func)
    def wrapper(arg):
        try:
            num = int(arg)
            return func(num)
        except ValueError as ve:
            return print("The input was not a valid integer.")
         
    return wrapper

@type_of_argument
@Negative_int
def factorial(num):
    if num == 0 or num == 1:
        return 1
    else:

        return num * factorial(num - 1)
    
num = input("Enter a number to find the factorial: ")
print(factorial(num))


# In[10]:


#2.	Write a class decorators to print the time required to execute a program.


# In[23]:


import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(num):
        start_time = time.perf_counter()    
        value = func(num)
        end_time = time.perf_counter()      
        run_time = end_time - start_time    
        print(f"Finished {func} in {run_time:.4f} secs")
        return value
    return wrapper_timer

@timer

def factorial(num):
    if num == 0 or num == 1:
        return 1
    else:
        return num * factorial(num - 1)
num = int(input("Enter a number to find the factorial: "))
a = factorial(num)
print(a)


# In[16]:


#3.	Write a decorator to multiply the output of the function by variable number. E.g
#a.	@multiply(3) over the function will multiply the output of function by 3
#b.	@multiply(5) over the same function will multiply the output of function by 5


# In[24]:


def multiplication(n):
    def wrapper(func):

        def wrapper(num):
            return func(num) * n

        return wrapper
    return wrapper

n = int(input("Enter a number to multiply the function output: "))
@multiplication(n)
def factorial(num):
    if num == 0 or num == 1:
        return 1
    else:
        return num * factorial(num - 1)
num = int(input("Enter a number to find the factorial: "))
print(factorial(num))


# In[25]:


#4.	Decorator a function that prints welcome message with 2 decorators
#@star
#@hash

#Output -
#***********************************
###################################
#Welcome to the world of Decorators
###################################
#***********************************


# In[19]:


def star(func):
    def wrapper(message):
        b = len(message)+ 1
        aa = b
        while (aa>=0):
            if aa>1:
                print("*",end = '')
            else:
                print()
            aa -= 1
        func(message)
        aa = b
        while (aa>=0):
            if aa>1:
                print("*",end = '')
            else:
                print()
            aa -= 1
    return wrapper

def hash(func):
    def wrapper(message):
        c = len(message) +1
        ad = c
        while (ad>=0):
            if ad>1:
                print("#",end = '')
            else:
                print()
            ad -= 1
        print(func(message))
        print()
        ad = c
        while (ad>=0):
            if ad>1:
                print("#",end = '')
            else:
                print()
            ad -= 1
    return wrapper


@star
@hash
def welcome_func(message):
    return message
a = "Welcome to the world of Decorators"
welcome_func(a)


# In[ ]:





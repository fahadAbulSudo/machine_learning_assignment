#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_5

# In[1]:


#1.	Define a class which has at least two methods:
#get_string: to get a string from console input
#print_string: to print the string in upper case.


# In[2]:


class StringWorks:
    def get_string(self):
        self.Input = str(input("Enter a string value : "))

    def print_string(self):
        print(self.Input)
        
    def swap_case(self):
        print (self.Input.swapcase())

obj = StringWorks()
obj.get_string()
obj.print_string()
obj.swap_case()


# In[3]:


#2.	Define a class named American and its subclass NewYorker. 
#Hints: Use class Subclass(ParentClass) to define a subclass.


# In[7]:


class Indian:
    
    def __init__(self, Nationality):
        self.Nationality = Nationality
        print("Initializing class")

    def details(self):
        return f"I am an {self.Nationality}"


class Maharashtrian(Indian):
    def __init__(self, Nationality, city):
        self.city = city
        print("Initializing subclass")    
        
        Indian.__init__(self, Nationality)  #Important: This line is used to use the Nationality field(Variable) of Parent Class(Indian)
    def details(self):
        return f"I am {self.Nationality} living in {self.city}"

A = Indian("Indian")
print(A.details())
B = Maharashtrian("Indian", "Mumbai")
print(B.details())


# In[ ]:


#3.	Write a program to illustrate Append, Delete and Display Elements of a List using classes.


# In[15]:


class ListOperations:
    def __init__(self):
        self.ListA = []

    def ListA_append(self, val):
        self.ListA.append(val)

    def ListA_delete(self, val):
        self.val = val
        if len(self.ListA) == 0 or (self.val not in self.ListA):
            print("The value is not in the list")
        else:
            self.ListA.remove(val)

    def ListA_display(self):
        if len(self.ListA) == 0:
            print("List is empty")
        else:
            print(self.ListA)


class Invalid_Choice_Exception(Exception):
    def __init__(self, choice):
        self.choice = choice

    def __str__(self):
        return f"{self.choice} is not a valid choice"
import sys


flag = "1"
lis = ListOperations()
while flag != "0":
    try:
        print("Please enter the choices for which functions do you want to run")
        choice = int(input("Enter 1, 2 or 3 for append, delete and display operations on list respectively: "))
    except Exception as e:
        print(e)
        sys.exit()
    if choice == 1:
        print("If you want to stop loop please enter 0")
        while True:
            try:
                val = int(input("Enter the number: "))
                if val == 0:
                    break
        
            except ValueError:
                print("No valid integer! Please try again ...")
            else:
                lis.ListA_append(val)
    elif choice == 2:
        try:
            val = int(input("Please enter an integer for deleting: "))
            val = int(val)
            lis.ListA_delete(val)
        except ValueError:
            print("No valid integer! Please try again ...")        
    elif choice==3:
        lis.ListA_display()

    else:
        result = None
        print(result)
        raise Invalid_Choice_Exception(choice)
    flag = input("to break loop enter '0' or not to stop enter '1': ")


# In[ ]:





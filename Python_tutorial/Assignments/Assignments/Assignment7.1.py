#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_7

# In[1]:


#1.	Write a program that reads a file and calculate the number of letters and digits?


# In[10]:


alphabets_count = 0
digits_count = 0
with open(r'C:\Users\abulf\Downloads\gtool.txt') as f:
    a = f.readlines()
    for i in str(a):
        if i.isalpha():
            alphabets_count += 1
        elif i.isdigit():
            digits_count +=1
print(f"No. of Letters: {alphabets_count}", )
print(f"No. of digits: {digits_count}", )            


# In[11]:


#2.	Write a program for a login system that takes username and password from the user. Save the username, password and profile details in a yaml file on sign up and then read the same file for sign in? 
#(Note: Add proper validations and try to design a scalable model in yaml file).


# In[7]:


from yaml import Loader
import yaml
def sign_up():
    username = input("Enter username: ")
    with open("Userdata.yaml", mode="r+") as file:
        data = yaml.load(file, Loader)
        if data and username == data.get("Username"):
            return print("Account with this username already exists")
        else:
            password = input("Enter password: ")
            name = input("Enter name: ")
            phone_no = input("Enter phone number: ")
            dic = {}
            dic["Username"] = username
            dic["Password"] = password
            dic["Name"] = name
            dic["Phone_no."] = phone_no 
            result = yaml.dump(dic, file, default_flow_style=False)
            #print(result)
                #file.seek(0)
            return print("Account created succesfully")
        
def sign_in():
    username = input("Enter username: ")
    password = input("Enter password: ")
    with open("Userdata.yaml", mode="r") as file:
        data = yaml.load(file, Loader)
        print(data)
        if data and username == data.get("Username"):
            if password == data.get("Password"):
                return print("Logged in successfully")
            else:
                return print("Incorrect password")
        elif data is None:
            return print("Please signup")
        else:
            return print("Incorrect username")


print("For sighnup enter A else B")
choice = input()
if choice == 'A':
    sign_up()
elif choice == 'B':
    sign_in()
else:
    print("please enter correct choice")


# In[ ]:


#3.	Write a program to dump a dictionary to a json file and then load the json file to another dictionary?


# In[ ]:


import json as js

details = {'Email':'Fahad@gmail.com', 'Phone_no.':93730}
Dic = {}
with open('json_file.json', 'w') as f:
    js.dump(details, f)

with open('json_file.json', 'r') as f:
    Dic = js.load(f)
    print(Dic)


# In[ ]:


#4.	Write a program to create a class which performs Basic Calculator Operations.


# In[2]:


class Calculator:
    def __init__(self):
        self.num1 = 0
        self.num2 = 0

    def add(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        return self.num1 + self.num2

    def multiply(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        return self.num1 * self.num2
    
    def divide(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        try:
            return self.num1 / self.num2
        except Exception as e:
            return e

    def subtract(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        return self.num1 - self.num2

    def remainder(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        try:
            return self.num1 % self.num2
        except Exception as e:
            return e
        
    def exponential(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        return self.num1 ** self.num2
print("1 - Addition")
print("2 - Mutiplication")
print("3 - Division")
print("4 - Substraction")
print("5 - Remainder")
print("6 - Exponential")
print("0 - None")
flag = 1
obj = Calculator()
class Invalid_Choice_Exception(Exception):
    def __init__(self, choice):
        self.choice = choice

    def __str__(self):
        return f"{self.choice} is not a valid choice"
import sys
while flag != 0:
    try:
        choice = int(input("Enter the number from above options:"))
    except Exception as e:
        print(e)
        sys.exit()
    if choice == 1:
        num1 = float(input("Enter number 1: "))
        num2 = float(input("Enter number 2: "))
        print(obj.add(num1, num2))
    elif choice == 2:
        num1 = float(input("Enter number 1: "))
        num2 = float(input("Enter number 2: "))
        print(obj.multiply(num1, num2))
    elif choice == 3:
        num1 = float(input("Enter number 1: "))
        num2 = float(input("Enter number 2: "))
        print(obj.divide(num1, num2))
    elif choice == 4:
        num1 = float(input("Enter number 1: "))
        num2 = float(input("Enter number 2: "))
        print(obj.subtract(num1, num2))
    elif choice == 5:
        num1 = int(input("Enter number 1: "))
        num2 = int(input("Enter number 2: "))
        print(obj.remainder(num1, num2))
    elif choice == 6:
        num1 = float(input("Enter number 1: "))
        num2 = float(input("Enter number 2: "))
        print(obj.exponential(num1, num2))
    elif choice == 0:
        flag = 0
    else:
        result = None
        print(result)
        raise Invalid_Choice_Exception(choice)


# In[31]:


#5.	Write a program that will merge multiple files into a single file line by line
#(Hint: Consider all the corner cases and files can be large in size)?


# In[32]:


data1 = ""
data2 = "" 
with open('Important notes made by me fa.txt') as file1:
    data1 = file1.read()
  
with open('TimeStamps.txt') as file2:
    data2 = file2.read()
    
data1 += "\n"
data1 += "\n"
data1 += data2
  
with open ('file3.txt', 'w') as fp:
    fp.write(data1)


# In[33]:


#6.	Write a program to read a file and capitalize the First Letter of every word in the file?


# In[34]:



result = ""    
with open('file3.txt', 'r+') as f:
    data = f.read()
    list_of_words = data.split()
    for elem in list_of_words:
        if len(result) > 0:
            result = result + " " + elem.strip().capitalize()
        else:
            result = elem.capitalize()
    data = result
    print(data)
    f.seek(0)
    f.write(data)


# In[35]:


#7.	Write a program that reads a text file and counts the number of times a certain letter appears in the text file.


# In[36]:


letter = input("Enter the letter to be counted: ")
letter = letter.lower()
count = 0
with open('file3.txt', 'r') as f:
    data = f.read()
    data = data.lower()
    for a in data:
        if a==letter:
            count+=1
print(count)


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT_2

# In[3]:


#1.	Write a password generator program of 8 characters that contains at least 
#one uppercase, one lower case, one digit and one special symbol?
#(Every new password should be different than the previously generated one).


# In[21]:



import random
pevious_password = ""
def password_generator():
    print('password_generator()')
    while True:

        MAX_LEN = 8
 
        DIGIT = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
        LOCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                     'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q',
                     'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                     'z']
 
        UPCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                     'I', 'J', 'K', 'M', 'N', 'O', 'P', 'Q',
                     'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                     'Z']
 
        SYMBOL = ['@', '#', '$', '%', '=', ':', '?', '.', '/', '|', '~', '>',
           '*', '(', ')', '<']
 
        COMBINED = DIGIT + UPCASE + LOCASE + SYMBOL
 
        random_digit = random.choice(DIGIT)
        random_upper = random.choice(UPCASE)
        random_lower = random.choice(LOCASE)
        random_symbol = random.choice(SYMBOL)

        temp = random_digit + random_upper + random_lower + random_symbol
 
 

        for x in range(MAX_LEN - 4):
            temp = temp + random.choice(COMBINED)
 
        #code to covert string character wise to list random.shuffle() method only take list ass arguments 
        list1=[]
        list1[:0]=temp    
        #shuffle the list
        random.shuffle(list1)
        print(temp)
        print(list1)
        # traverse list to string
        password = ""
        for x in list1:
            password = password + x
    
        global pevious_password
        if  pevious_password is password:
            continue
        else:
            pevious_password = password
            yield password     
G = password_generator()
G
#print(password)


# In[22]:


next(G)


# In[23]:


next(G)


# In[24]:


next(G)


# In[25]:


#2.	Write a function to reverse words of the sentence?


# In[29]:


def reverse(Sentence):
    #Create an empty character array stack
    reverse_ = []
    words = Sentence.split(" ")
     
    # Push words into the stack
    for word in words :
        reverse_.append(word);
         
    while (len(reverse_)) :
        
        print(reverse_.pop(), end = " ");
     
 
Sentence = input()
reverse(Sentence)


# In[1]:


#3.	Write a function that accepts a sequence of whitespace separated words
#as input and prints the words after removing all duplicate words and sorting them alphanumerically?
#(Both with and without second list)


# In[12]:


s = input()
s = s.lower()# lower al words to make uniform
lis1 = s.split(" ")# split and srored every word in lis1
print(lis1)
se = set(lis1) #convert list to set so that duplicates are not allowed
se=sorted(se) #sorted the set and converted to list
print(se)
str1 = ""
for ele in se: 
    str1 += (ele+" ")#here ry to convert list to set
#print(str1.join(se))
print(str1)


# In[24]:


s = input()
#words = [word for word in s.split(" ")]
s = s.lower()
words = s.split(" ")
print (" ".join(sorted(set(words))))


# In[25]:


#4.	Given an array of ints length n, return an array with the elements "rotated left" so {1, 2, 3} yields each iteration until {2, 3, 1} comes. Eg:
#rotate_left1([5, 11, 9]) → [11, 9, 5]
#rotate_left2([7, 0, 0]) → [0, 7, 0]	


# In[39]:


#C language type
import copy
def left_rotation(a):
        
        for l in range(2):    
            #j, last    
            
            first = a[0]    
        
            for j in range(1,n):    
            #Shift element of array by one    
                a[j-1] = a[j] 
            a[n-1]= first
            yield a
            
a = []
b = []
n = int(input("Enter number of elements : "))
  

for o in range(0, n):
    elem = int(input())
    a.append(elem)
b = copy.copy(a) 
a = left_rotation(a)
for l in range(2):
    print(f"rotate_left{l+1} of  {b}  gives {next(a)}")


# In[48]:


#python method of slicing
def left_rotation(a,n):  #n,d
    temp = []
    k = 0
    while (k < 2):
        temp.append(a[k])
        k = k + 1
    k = 0
    d = 2
    while (d < n):
        a[k] = a[d]
        k = k + 1
        d = d + 1
    a[:] = a[: k] + temp
    return a
a = []

n = int(input("Enter number of elements : "))
  

for o in range(0, n):
    elem = int(input())
    a.append(elem)
a = left_rotation(a,n)
print(f"After two rotations{a}")


# In[49]:


#5.	Write a function that takes the path of a directory and prints out the paths files within that directory 
#as well as any files present in the nested directories. (This function is similar to os.walk.
#Please don't use os.walk in your answer. We are interested in your ability to work with nested structures)


# In[10]:


import os

files = []
directories = []

def listdirs(rootdir):
    for folder in os.listdir(rootdir):
        d = os.path.join(rootdir, folder)
        if os.path.isdir(os.path.join(rootdir, folder)):
            directories.append(folder)
            listdirs(d)
        else:
            files.append(folder)
 
rootdir = r'C:\Users\abulf\Downloads'
listdirs(rootdir)
print(f"Directory list {directories}")
print(f"Files list {files}")


# In[ ]:






"""Write a package that will contain the modules for the following functionality
To return the common characters in the two strings.
Count the frequency of words appearing in the string
To return a new string made of the First 2 and Last 2 characters from a given string
Program to accept a Hyphen separated sequence of words as Input 
and return the Words in a hyphen separated sequence after sorting them alphabetically

Now, create a driver(can be a menu-driven) program that will call this package depending upon the functionality. 
Also, implement the exception handling by creating the custom exception.
"""

class Invalid_Choice_Exception(Exception):
    def __init__(self, choice):
        self.choice = choice

    def __str__(self):
        return f"{self.choice} is not a valid choice"
import sys
from Package_for_Q3 import Module3 as M

try:
    print("Please enter the choices for which functions do you want to run")
    choice = int(input("Enter 1 , 2 , 3 or 4: "))
except Exception as e:
    print(e)
    sys.exit()

if choice == 1:
    str1 = input("please enter string1: ")
    str2 = input("please enter string2: ")
    str1 = str1.split(" ")
    sr = ""
    str1 = sr.join(str1)
    str2 = str2.split(" ")
    se = ""
    str2 = se.join(str2)    
    result = M.intersecting_characters(str1, str2)
    print(result)
elif choice == 2:
    str1 = input("Enter the string: ")
    result = M.freq(str1)
    print(result)
elif choice==3:
    str = input("Enter the string: ")
    result = M.merge(str)
    print(result)
elif choice==4:
    str1  = input("Enter hyphen seperated string: ")
    result = M.hyphen(str1)
    print(result)
else:
    result = None
    print(result)
    raise Invalid_Choice_Exception(choice)

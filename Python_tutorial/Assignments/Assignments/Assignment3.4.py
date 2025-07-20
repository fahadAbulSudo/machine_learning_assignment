"""Create a package named “services” and perform the following operation
Add a sub-package by name - “math” and create module inside it for the following operations:
Functions to return the square root and cube root
Function to return the fibonacci series
Add a sub-package by name - “sorting” and create the modules for the following functionality:
Implement binary sort
Implement merge sort
A module to merge two list and return the sorted list(Use the merge sort from sorting package).

Now, create a module that will accept a list of numbers as an input. 
Create a new list having the square of the input list using the functions of math package and call the merge_sort method from point c and print a sorted list.
"""

from Services.pkj import Service_Merging
from Services.pkj import Squaring


e = []
  
n = int(input("Enter number of elements : "))
  

for o in range(0, n):
    ele = int(input())
    e.append(ele)
print(Squaring.squaring(e))

#to merger two lists and return a sorted list
list1 = e
list2 = Squaring.squaring(e)

result = Service_Merging.merge(list1, list2)
print(result)
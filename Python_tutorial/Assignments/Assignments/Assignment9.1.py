#Q1

import subprocess

patterns_file = 'patterns.txt'
readfile = 'romeo-full.txt'

with open(patterns_file, 'r') as f:
    for pattern in f:
        pattern = pattern.strip()

        process = subprocess.run(
            ['grep', '-c', f'{pattern}', readfile], capture_output=True, text=True)

        if int(process.stdout) == 0:
            print(
                f'The pattern "{pattern}" did not match any line of {readfile}')

            continue

        print(f'The pattern "{pattern}" matched {process.stdout.strip()} times')

#Q2

import time # import time module  
import threading  
from threading import *  
def cal_sqre(num): # define a square calculating function  
    print(" Calculate the square root of the given number")  
    for n in num: # Use for loop   
        time.sleep(0.3) # at each iteration it waits for 0.3 time  
        print(' Square is : ', n * n)  
  
def cal_cube(num): # define a cube calculating function  
    print(" Calculate the cube of  the given number")  
    for n in num: # for loop  
        time.sleep(0.3) # at each iteration it waits for 0.3 time  
        print(" Cube is : ", n * n *n)  
  
ar = [4, 5, 6, 7, 2] # given array  
  
t = time.time() # get total time to execute the functions  
#cal_cube(ar)  
#cal_sqre(ar)  
th1 = threading.Thread(target=cal_sqre, args=(ar, ))  
th2 = threading.Thread(target=cal_cube, args=(ar, ))  
th1.start()  
th2.start()  
th1.join()  
th2.join()  
print(" Total time taking by threads is :", time.time() - t) # print the total time  
print(" Again executing the main thread")  
print(" Thread 1 and Thread 2 have finished their execution.")  

#Q3

import subprocess
import os
with open(os.devnull, "wb") as limbo:
        for n in range(1, 10):
                ip="192.168.0.{0}".format(n)
                result=subprocess.Popen(["ping", "-c", "1", "-n", "-W", "2", ip],
                        stdout=limbo, stderr=limbo).wait()
                if result:
                        print (f"{ip} is inactive")
                else:
                        print (f" {ip} Is active ")

#Q4a


# Python3 program to check 
# string is alphanumeric or
# not using Regular Expression. 

import re

def isAlphaNumeric(str):
    # Regex to check string is alphanumeric or not.
    regex = "^(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9]+$"
    p = re.compile(regex)
    if(str == None):
        return False
    if(re.search(p, str)):
        return True
    else:
        return False

str1 = "Afour123"
print(str1, ":", isAlphaNumeric(str1))

#Q4b

import re
 
# Function that checks if a string 
# contains uppercase, lowercase 
# special character & numeric value 

def isAllPresent(str):
    regex = ("^(?=.*[a-z])(?=." +
             "*[A-Z])(?=.*\\d)" +
             "(?=.*[-+_!@#$%^&*., ?]).+$")
    p = re.compile(regex)
    if (str == None):
        print("No")
        return  
    if(re.search(p, str)):
        print("Yes")
    else:
        print("No")

str = "Afour123@"
isAllPresent(str)

#Qc
import re 
# Make a regular expression
# for validating an Ip-address
regex = "^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])$"
def check(Ip): 
    if(re.search(regex, Ip)): 
        print("Valid Ip address") 
    else: 
        print("Invalid Ip address")  
# Driver Code 

Ip = "192.168.0.1"
check(Ip)

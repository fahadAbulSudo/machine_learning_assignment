'''2.	Write a basic program with the help of modules and packages for Bank services 
(Deposit and Withdraw money) with proper validation.'''



from Banking import Banking_method as BK
  

Acc = BK.Bank_Account()
  
# Calling functions with that class object
Acc.deposit()
Acc.withdraw()
Acc.display()
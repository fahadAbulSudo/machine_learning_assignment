import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('insurance.csv')
print(df.isnull().sum())
print(df.isnull())
print(df.head())
#sex
le = LabelEncoder()
le.fit(df.sex) 
df.sex = le.transform(df.sex)
# smoker or not
le.fit(df.smoker) 
df.smoker = le.transform(df.smoker)
#region
le.fit(df.region) 
df.region = le.transform(df.region)
print(df.region)
print(df.corr()['charges'])
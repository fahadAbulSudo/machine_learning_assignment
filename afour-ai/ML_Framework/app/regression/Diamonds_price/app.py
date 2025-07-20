import numpy as np
import warnings
import pickle
import os
warnings.filterwarnings("ignore")

abs = os.getcwd()
abs = abs.replace("\\app\\regression\\Diamonds_price","")

randomforest_model = pickle.load(open(abs + "\\output\\Regression\\diamonds_price\\randomforest_model.pkl", 'rb'))
cut_label_encoder = pickle.load(open(abs + "\\output\\Regression\\diamonds_price\\cut_label_encoder.pkl", 'rb')) 
color_label_encoder = pickle.load(open(abs + "\\output\\Regression\\diamonds_price\\color_label_encoder.pkl", 'rb'))
clarity_label_encoder = pickle.load(open(abs + "\\output\\Regression\\diamonds_price\\clarity_label_encoder.pkl", 'rb'))
standardize_diamond = pickle.load(open(abs + "\\output\\Regression\\diamonds_price\\standardize_diamond.pkl", 'rb'))
normalize_diamond = pickle.load(open(abs + "\\output\\Regression\\diamonds_price\\normalize_diamond.pkl", 'rb'))
price_diamond = pickle.load(open(abs + "\\output\\Regression\\diamonds_price\\price_diamond.pkl", 'rb'))

carat = float(input("Carat(max=5): "))
cut = (input("Cut['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']: "))
color = (input("Color['E', 'I', 'J', 'H', 'F', 'G', 'D']: "))
clarity = (input("Clarity['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']: "))
depth = float(input("Depth(max=70): "))
table = float(input("Table(max=90): "))
x = float(input("Dimension in X direction(max=10): "))
y = float(input("Dimension in Y direction(max=50): "))
z = float(input("Dimension in Z direction(max=30): "))
vol = x*y*z

cut_lst = []
cut_lst.append(cut)
color_code_lst = []
color_code_lst.append(color)
clar_lst = []
clar_lst.append(clarity)
cut= cut_label_encoder.transform(cut_lst)
color = color_label_encoder.transform(color_code_lst)
clarity = clarity_label_encoder.transform(clar_lst)

sca = []
sca.append(depth)
sca = np.reshape(sca, (-1,1 ))
sca1 = standardize_diamond.transform(sca)

nor = []
nor.append(carat)
nor.append(table)
nor.append(vol)
nor = np.reshape(nor, (-1,3 ))
nor1 = normalize_diamond.transform(nor)

main_lst = []
main_lst.append(nor1[0][0])
main_lst.append(cut[0])
main_lst.append(color[0])
main_lst.append(clarity[0])
main_lst.append(sca1[0][0])
main_lst.append(nor1[0][1])
main_lst.append(nor1[0][2])
main_lst = np.reshape(main_lst, (-1,7 ))

pred = randomforest_model.predict(main_lst)
pred = np.reshape(pred, (-1,1 ))
pred = price_diamond.inverse_transform(pred)
print(pred[0][0])
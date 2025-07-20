import streamlit as st
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")

employee_skill = pickle.load(open('../output/employee_skill.pkl', 'rb'))
similarity = pickle.load(open('../output/similarity.pkl', 'rb'))

'''
       This code is used to Recommend employee by taking employeeID as input 
       and using pickle model and dataset pickle from output folder 
'''

def recommend(employee_id):
    #This function is used to recommend employee list with their respective similarity
    #By using pickle models 
  employee_lst = []
  sim = []
  index = employee_skill[employee_skill['employee_id'] == employee_id].index[0]
  distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
  for i in distances[1:11]:
    employee_lst.append(employee_skill.employee_id[i[0]])
    sim.append(i[1])
  sim = [x * 100 for x in sim]
  sim = [ '%.2f' % elem for elem in sim ]
  lst = list(zip(employee_lst, sim))

  return lst

def mapping_desig(val):
    #This function is used to compare input employee designation with recommended employee designation
    mask = employee_skill['employee_id'].values == employee
    df_new = employee_skill.loc[mask]
    val_desig = df_new.designation_name.values[0]
    if val == val_desig:
        return 1
    else:
        return 0

def mapping_depar(val):
    #This function is used to compare input employee department with recommended employee department
    mask = employee_skill['employee_id'].values == employee
    df_new = employee_skill.loc[mask]
    val_depar = df_new.department_name.values[0]
    if val == val_depar:
        return 1
    else:
        return 0


def sorted_dataframe(employee_lst, employee):
    #This function gives complete sorted and recommended dataframe
    #Below line code is used to unzip the list of employee recommend and similarity percentage 
    employee_lst = [[i for i, j in employee_lst],
       [j for i, j in employee_lst]]
    lst = employee_lst[0]
    sim = employee_lst[1]
    sim.append(0)
    p = len(sim)-1
    while(p>0):
        sim[p] = sim[p-1]
        p = p - 1
    sim[0] = 0
    mask = employee_skill['employee_id'].values == employee
    df_new = employee_skill.loc[mask]
    for employee1 in lst:
        mask = employee_skill['employee_id'].values == employee1
 
# using loc() method
        df_new1 = employee_skill.loc[mask]
        df_new = df_new.append(df_new1, ignore_index = True) 
    df_new2 = df_new.loc[1:]
    df_new['similarity'] = sim
    b = df_new2.iloc[:,5:6].values
    b = b.reshape(-1)
    b=b.tolist()
    encoded_de = []
    for value in b:
        encoded_de.append(mapping_depar(value))
    df_new2['department_name'] = encoded_de
    b=df_new2.iloc[:,4:5].values
    b = b.reshape(-1)
    b=b.tolist()
    encoded_category = []
    for category in b:
        if category == "EMPLOYEE":
            encoded_category.append(2)
        elif category == "Intern" or category == "CONTRACT":
            encoded_category.append(1)
        else:
            encoded_category.append(0)

    df_new2['category'] = encoded_category
    
    mask = employee_skill['employee_id'].values == employee
    df_new6 = employee_skill.loc[mask]
    #print(df_new1)
    coe = df_new6.coe.values[0]
    del df_new6
    b=df_new2.iloc[:,9:10].values
    b = b.reshape(-1)
    b=b.tolist()
    
    encoded_coe = []
    if len(coe) == 2:
        for val in b:
            if len(val) == 2:
                encoded_coe.append(2)
            elif val[0] == 'DESIGN_DEVELOPMENT' or val[0] == 'RELIABILITY_ENGINEERING':
                encoded_coe.append(1)
            else:
                encoded_coe.append(0)
    elif coe[0] == "DESIGN_DEVELOPMENT":
        for val in b:
            if len(val) == 2 or val[0] == "DESIGN_DEVELOPMENT":
                encoded_coe.append(2)
            else:
                encoded_coe.append(0)
    elif coe[0] == "RELIABILITY_ENGINEERING":
        for val in b:
            if len(val) == 2 or val[0] == "RELIABILITY_ENGINEERING":
                encoded_coe.append(2)
            else:
                encoded_coe.append(0)
    else:
        for val in b:
            if len(val) == 2:
                encoded_coe.append(2)
            elif len(val) == 1:
                encoded_coe.append(1)
            else:
                encoded_coe.append(0)

    df_new2['coe'] = encoded_coe

    b=df_new2.iloc[:,10:11].values
    b = b.reshape(-1)
    b=b.tolist() 

    name_recommend = []
    for val in b:
        if "Bench" in val:
            name_recommend.append(1)
        else:
            name_recommend.append(0)

    df_new2['name'] = name_recommend
    b=df_new2.iloc[:,6:7].values
    b = b.reshape(-1)
    b=b.tolist()

    encoded_hierarchy = []
    for value in b:
        encoded_hierarchy.append(mapping_desig(value))

    df_new2['designation_name'] = encoded_hierarchy
    
    mask = employee_skill['employee_id'].values == employee
    df_new4 = employee_skill.loc[mask]
    duheadid = df_new4.duhead_id.values[0]
    del df_new4

    b=df_new2.iloc[:,7:8].values
    b = b.reshape(-1)
    b=b.tolist()

    du_recommend = []
    for val in b:
        lst5 = []
        for heads in val:
            if heads in duheadid:
                lst5.append(1)
            else:
                lst5.append(0)

        if 1 in lst5:
            du_recommend.append(1)
        else:
            du_recommend.append(0)

    df_new2['duhead_id'] = du_recommend

    mask = employee_skill['employee_id'].values == employee
    df_new4 = employee_skill.loc[mask]
    coeheadid = df_new4.coehead_id.values[0]
    del df_new4

    b=df_new2.iloc[:,8:9].values
    b = b.reshape(-1)
    b=b.tolist()

    co_recommend = []
    for val in b:
        lst5 = []
        for heads in val:
            if heads in coeheadid:
                lst5.append(1)
            else:
                lst5.append(0)

        if 1 in lst5:
            co_recommend.append(1)
        else:
            co_recommend.append(0)

    df_new2['coehead_id'] = co_recommend
    mask = employee_skill['employee_id'].values == employee
    df_new6 = employee_skill.loc[mask]
    worklocation = df_new6.work_location.values[0]
    del df_new6

    b=df_new2.iloc[:,11:12].values
    b = b.reshape(-1)
    b=b.tolist()

    work_en = []
    for val in b:
        lst5 = []
        for locations in val:
            if locations in worklocation:
                lst5.append(1)
            else:
                lst5.append(0)

        if 1 in lst5:
            work_en.append(1)
        else:
            work_en.append(0)

    df_new2['work_location'] = work_en
    
    b=df_new2.iloc[:,12:13].values
    b = b.reshape(-1)
    b=b.tolist()

    encoded_pt = []
    for val in b:
        if len(val) == 2 or val[0] == "CUSTOMER":
            encoded_pt.append(0)
        elif val[0] == 'ASSET':
            encoded_pt.append(1)
        else:
            encoded_pt.append(1)
    df_new2['project_type'] = encoded_pt

    df_new3 = df_new2[['category', 'department_name',
        'designation_name', 'duhead_id', 'coehead_id', 'coe',
       'name', 'work_location', 'project_type']].copy()

    listOfDFRows = df_new3.to_numpy().tolist()

    sum = []
    for val in listOfDFRows:
        count = 0
        for i in val:
            count = count + i
        sum.append(count)

    df_re = df_new.loc[1:]
    df_re['Imporance_value'] = sum
    res_df = df_re.sort_values(by = 'Imporance_value', ascending = False)
    res_df.drop(columns=['duhead_id', 'coehead_id', 'name',
        'allocation_percentage','Imporance_value'], axis=1, inplace=True)
    res_df.loc[res_df["coe"] == 'no_info', "coe"] = "--"
    res_df.loc[res_df["work_location"] == 'no_info', "work_location"] = "--"
    res_df.loc[res_df["project_type"] == 'no_info', "project_type"] = "--"
    res_df.loc[res_df["designation_name"] == "no_info", "designation_name"] = "--"
    res_df.loc[res_df["category"] == "no_info", "category"] = "--"
    res_df['department_name'] = res_df['department_name'].fillna("--")
    res_df['first_name'] = res_df['first_name'].fillna("--")
    res_df['last_name'] = res_df['last_name'].fillna("--")
    res_df = res_df[['employee_id', 'first_name', 'last_name', 'category',
       'department_name', 'designation_name', 'coe',
        'work_location', 'project_type', 'similarity', 'employee_skills']]
    res_df.set_index("employee_id", inplace = True)
    
    return res_df 

def print_dataframe(employee):
    mask = employee_skill['employee_id'].values == employee
    df = employee_skill.loc[mask]
    df.drop(columns=['duhead_id', 'coehead_id', 'name',
        'allocation_percentage'], axis=1, inplace=True)
    df = df[['employee_id', 'first_name', 'last_name', 'category',
       'department_name', 'designation_name', 'coe',
        'work_location', 'project_type', 'employee_skills']]
    df.set_index("employee_id", inplace = True)
    return df

st.title('Welcome')
st.subheader("Employee recommendation App")
with st.form(key='recommend'):
    employee = st.selectbox(

    "Type or select employee id from the dropdown",

    employee_skill['employee_id'].values

    )
    submit_button = st.form_submit_button(label='Recommend')

if submit_button:
  employee_list = recommend(employee)
  employee_dataframe = sorted_dataframe(employee_list, employee)
  main_employee_dataframe = print_dataframe(employee)
  st.dataframe(main_employee_dataframe)
  st.dataframe(employee_dataframe)
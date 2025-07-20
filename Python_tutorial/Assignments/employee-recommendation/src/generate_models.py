def generate_models():
       from sklearn.feature_extraction.text import CountVectorizer
       from sklearn.metrics.pairwise import cosine_similarity
       import pandas as pd
       import numpy as np
       import warnings
       import pickle
       import os
       warnings.filterwarnings("ignore")
       cv = CountVectorizer(max_features=10000,stop_words='english')
       employee = pd.read_csv(os.path.abspath("input/employee.csv"))
       project_resource = pd.read_csv(os.path.abspath("input/employeeprojectresource.csv"))
       project = pd.read_csv(os.path.abspath("input/project.csv"))
       department = pd.read_csv(os.path.abspath("input/department.csv")) 
       designation = pd.read_csv(os.path.abspath("input/designation.csv")) 
       employee_skills = pd.read_csv(os.path.abspath("input/employeeskill.csv"))
       skills =  pd.read_csv(os.path.abspath("input/skills.csv"))

       employee = pd.merge(employee, department, on='department_id', how='outer')
       employee = pd.merge(employee, designation, on='designation_id', how='left')
       employee.drop(columns=['_id_x', 'commutation', 'date_of_birth', 'date_of_joining', 'desk',
              'email', 'id_card_no', 'image_url',
              'status', 'department_id', 'designation_id', 'du_head_id',
              'manager_id', 'coe', 'date_of_confirmation', 'location',
              '_id_y', '_id'], axis=1, inplace=True)

       employee.category = np.where(employee.employee_id.str.startswith("AFT0"), employee.category.ffill(), employee.category)
       employee.category = np.where(employee.employee_id.str.startswith("AFTP"), employee.category.ffill(), employee.category)
       employee.category = np.where(employee.employee_id.str.startswith("AFTI"), employee.category.ffill(), employee.category)
       employee["category"].fillna("EMPLOYEE", inplace = True)

       employee = employee.dropna(axis=0, subset=['designation_name'])
       employee = employee.dropna(axis=0, subset=['department_name'])

       project_resource.drop(columns=['_id', 'project_resource_id', 'is_active',
              'reporting_manager', 'start_date', 'end_date',
              ], axis=1, inplace=True)

       project.drop(columns=['actual_end_date', 'actual_start_date',
              'cost_center', 'description', 'engagement_type', 'expected_end_date',
              'expected_start_date', 'services_type', 'status',
              'sub_services_type', 'service_type'], axis=1, inplace=True)

       project = pd.merge(project_resource, project, on='project_id', how='left')
       employee_project = pd.merge(employee,project, on='employee_id', how='left')
       employee_project.drop(columns=['project_id'], axis=1, inplace=True)
       employee_project = employee_project.drop_duplicates(
        subset = ['employee_id', 'name'],
        keep = 'last').reset_index(drop = True)

       df_headids = employee_project[['employee_id', 'duhead_id']].copy()
       df_headids['duhead_id'].fillna(df_headids['duhead_id'].mode()[0], inplace=True)
       df_headids.drop_duplicates(inplace=True)
       df_headids = df_headids.groupby('employee_id')['duhead_id'].apply(list).reset_index(name='duhead_id')
       employee = pd.merge(employee, df_headids, on='employee_id', how='left')

       df_headids = employee_project[['employee_id', 'coehead_id']].copy()
       df_headids['coehead_id'].fillna(df_headids['coehead_id'].mode()[0], inplace=True)
       df_headids.drop_duplicates(inplace=True)
       df_headids = df_headids.groupby('employee_id')['coehead_id'].apply(list).reset_index(name='coehead_id')
       employee = pd.merge(employee, df_headids, on='employee_id', how='left')

       df_coe = employee_project[['employee_id', 'coe']].copy()
       df_coe['coe'].fillna(df_coe['coe'].mode()[0], inplace=True)
       df_coe.drop_duplicates(inplace=True)
       df_coe = df_coe.groupby('employee_id')['coe'].apply(list).reset_index(name='coe')
       employee = pd.merge(employee, df_coe, on='employee_id', how='left')

       df_name = employee_project[['employee_id', 'name']].copy()
       df_name["name"].fillna("Bench", inplace=True)
       df_name.drop_duplicates(inplace=True)
       df_name = df_name.groupby('employee_id')['name'].apply(list).reset_index(name='name')
       employee = pd.merge(employee, df_name, on='employee_id', how='left')

       df_WL = employee_project[['employee_id', 'work_location']].copy()
       df_WL['work_location'].fillna(df_WL['work_location'].mode()[0], inplace=True)
       df_WL.drop_duplicates(inplace=True)
       df_WL = df_WL.groupby('employee_id')['work_location'].apply(list).reset_index(name='work_location')
       employee = pd.merge(employee, df_WL, on='employee_id', how='left')

       df_pt = employee_project[['employee_id', 'project_type']].copy()
       df_pt['project_type'].fillna(df_pt['project_type'].mode()[0], inplace=True)
       df_pt.drop_duplicates(inplace=True)
       df_pt = df_pt.groupby('employee_id')['project_type'].apply(list).reset_index(name='project_type')
       employee = pd.merge(employee, df_pt, on='employee_id', how='left')

       df_all = project_resource[['employee_id', 'allocation_percentage']].copy()
       df_all['allocation_percentage'] = df_all['allocation_percentage'].fillna(0)
       df_all['allocation_percentage'] = df_all['allocation_percentage'].astype(int)
       df_all1 = df_all.groupby('employee_id', as_index =False)['allocation_percentage'].sum()
       employee = pd.merge(employee, df_all1, on='employee_id', how='left')

       employee['allocation_percentage'] = employee['allocation_percentage'].fillna(0)
       employee['allocation_percentage'] = employee['allocation_percentage'].astype(int)
       merged_employee_skills_and_skills = employee_skills.merge(skills, how='inner', left_on='skill_id', right_on='_id')
       final_merged_employee_skills = merged_employee_skills_and_skills.drop(columns=['_id_x','_id_y','status'],axis=1)
       final_merged_employee_skills = final_merged_employee_skills.replace(np.nan," ")
       final_merged_employee_skills['employee_skills'] = final_merged_employee_skills['sub_technology']+" "+final_merged_employee_skills['technology']+" "+final_merged_employee_skills['tool']+" "+final_merged_employee_skills['work_type']
       three_column_final_skills = final_merged_employee_skills.drop(columns = ['sub_technology', 'technology', 'tool',
              'work_type'],axis = 1)

       grp_emp = three_column_final_skills.groupby('employee_id')
       unique_employee_ids = three_column_final_skills.employee_id.unique()
       sorted_unique_employee_ids = sorted(unique_employee_ids)
       dictt = {'employee_id':[],'employee_skills':[]}
       employee_skill_dataframe = pd.DataFrame(dictt)
       for i in sorted_unique_employee_ids:
              skill_list = []
              for j in grp_emp.get_group(i)['employee_skills']:
                     skill_list.append(j)
              employee_skill_dataframe.loc[len(employee_skill_dataframe.index)] = [i,",".join(skill_list)] 

       employee =  pd.merge(employee_skill_dataframe, employee, how='left', on='employee_id')
       vector = cv.fit_transform(employee['employee_skills']).toarray()
       similarity = cosine_similarity(vector)

       employee['designation_name'] = employee['designation_name'].fillna('no_info')
       employee['coe'] = employee['coe'].fillna('no_info')
       employee['category'] = employee['category'].fillna('no_info')
       employee['duhead_id'] = employee['duhead_id'].fillna('no_info')
       employee['coehead_id'] = employee['coehead_id'].fillna('no_info')
       employee['name'] = employee['name'].fillna('no_info')
       employee['work_location'] = employee['work_location'].fillna('no_info')
       employee['project_type'] = employee['project_type'].fillna('no_info')
       employee['allocation_percentage'] = employee['allocation_percentage'].fillna(0)

       pickle.dump(employee, open(os.path.abspath('output/employee_skill.pkl'), 'wb'))
       pickle.dump(similarity, open(os.path.abspath('output/similarity.pkl'), 'wb'))
       

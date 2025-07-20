from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import re
import pickle

class Handler(FileSystemEventHandler):
    def on_moved(self, event):
        if event.src_path == ".\\employee_final.csv" and event.src_path == ".\\employee_skill_dataframe.csv":
            print("modified")
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=10000,stop_words='english')
            employee_skill = pd.read_csv("employee_final.csv")
            employee_skill.drop(employee_skill.columns[[0]], axis=1, inplace=True)

            vector = cv.fit_transform(employee_skill['employee_skills']).toarray()
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(vector)
            employee_skill.loc[employee_skill["coe_x"] == "['DESIGN_DEVELOPMENT', 'RELIABILITY_ENGINEERING']", "coe_x"] = "['RELIABILITY_ENGINEERING', 'DESIGN_DEVELOPMENT']"
            employee_skill['designation_name'] = employee_skill['designation_name'].fillna('no_info')
            employee_skill['coe_x'] = employee_skill['coe_x'].fillna("['no_info']")
            employee_skill['category_x'] = employee_skill['category_x'].fillna('no_info')
            employee_skill['duhead_id'] = employee_skill['duhead_id'].fillna("['no_info']")
            employee_skill['coehead_id'] = employee_skill['coehead_id'].fillna("['no_info']")
            employee_skill['name'] = employee_skill['name'].fillna("['no_info']")
            employee_skill['work_location'] = employee_skill['work_location'].fillna("['no_info']")
            employee_skill['project_type'] = employee_skill['project_type'].fillna("['no_info']")
            employee_skill.loc[employee_skill["project_type"] == "['CUSTOMER', 'ASSET']", "project_type"] = "['ASSET', 'CUSTOMER']"
            employee_skills = pd.read_csv("employee_skill_dataframe.csv")

            employee_skill.loc[employee_skill["coe_x"] == "['DESIGN_DEVELOPMENT', 'RELIABILITY_ENGINEERING']", "coe_x"] = "['RELIABILITY_ENGINEERING', 'DESIGN_DEVELOPMENT']"
            employee_skill['designation_name'] = employee_skill['designation_name'].fillna('no_info')
            employee_skill['coe_x'] = employee_skill['coe_x'].fillna("['no_info']")
            employee_skill['category_x'] = employee_skill['category_x'].fillna('no_info')
            employee_skill['duhead_id'] = employee_skill['duhead_id'].fillna("['no_info']")
            employee_skill['coehead_id'] = employee_skill['coehead_id'].fillna("['no_info']")
            employee_skill['name'] = employee_skill['name'].fillna("['no_info']")
            employee_skill['work_location'] = employee_skill['work_location'].fillna("['no_info']")
            employee_skill['project_type'] = employee_skill['project_type'].fillna("['no_info']")
            employee_skill.loc[employee_skill["project_type"] == "['CUSTOMER', 'ASSET']", "project_type"] = "['ASSET', 'CUSTOMER']"
            employee_skills = pd.read_csv("employee_skill_dataframe.csv")
            employee_skill = pd.merge(employee_skill, employee_skills, on='employee_id', how='left')
            employee_skill.drop(employee_skill.columns[[15]], axis=1, inplace=True)

            pickle.dump(employee_skill, open('employee_skill.pkl', 'wb'))
            pickle.dump(similarity, open('similarity.pkl', 'wb'))


observer = Observer()
observer.schedule(Handler(), ".")
observer.start()
try:
    while True:
        pass
except KeyboardInterrupt:
    observer.stop()
observer.join()

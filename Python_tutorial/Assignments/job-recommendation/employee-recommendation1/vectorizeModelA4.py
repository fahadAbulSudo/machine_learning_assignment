import re
import pickle
import pandas as pd
import numpy as np
import nltk
import re
import os
import string
import shutil
import time
from convertAndParse import convertAndParse
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#from collections import defaultdict
from nltk.corpus import stopwords
#from nltk.corpus import wordnet
# import boto3

def fileSupported(filename):
  supportFormats = [".docx", ".doc", ".dot", ".dotx", ".odt", ".pdf", ".txt", ".rtf"]
  supported = False
  for extension in supportFormats:
    if filename.endswith(extension) or filename.endswith(extension.upper()):
      supported = True
  return supported

def skill_clean(val):
    val = val.replace("\n", " ")
    val = val.replace("\t", " ")
    val = val.replace(",", " ")
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    val = _RE_COMBINE_WHITESPACE.sub(" ", val).strip()
    val = val.replace("/\d\.\s+|[a-z]\)\s+|•\s+|[A-Z]\.\s+|[IVX]+\.\s+/g", "")
    val = val.replace("skillset summary", "")
    val = val.replace("overall summary", "")
    val = val.replace("qualification", "")
    val = val.strip()
    val = re.sub(" \d+", " ", val)
    val = re. sub(r'[:]', '', val) 
    #val = re.sub(r'[^\w\s]', '', val)
    #val = word_tokenize(val)
    val = [word for word in val.split() if word.lower() not in sw_nltk]
    lemmatizer = WordNetLemmatizer()
    val = [lemmatizer.lemmatize(word) for word in val]
    #print(val)
    var = ""
    for v in val:
       var = var + " " + v
    return var

def response_clean(val):
    val = val.replace("\n", " ")
    val = val.replace("\t", " ")
    val = val.replace(",", " ")
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    val = _RE_COMBINE_WHITESPACE.sub(" ", val).strip()
    val = val.replace("/\d\.\s+|[a-z]\)\s+|•\s+|[A-Z]\.\s+|[IVX]+\.\s+/g", "")
    val = val.strip()
    val = re.sub(" \d+", " ", val)
    val = re. sub(r'[:]', '', val)
    #val = re.sub(r'[^\w\s]', '', val)
    #val = word_tokenize(val)
    val = [word for word in val.split() if word.lower() not in sw_nltk]
    lemmatizer = WordNetLemmatizer()
    val = [lemmatizer.lemmatize(word) for word in val]
    var = ""
    for v in val:
        var = var + " " + v
    return var

def skills_data(data):
    res1 = []
    res2 = []
    skl = re.findall('skill.*summary', data)
    if skl:
        if "overall summary" in data:
            regex_skills = re.compile(skl[0] + ".+?overall summary", flags=re.DOTALL)
            temp = re.findall(regex_skills, data)
            for val in temp:
                res1.append(skill_clean(val))

            if len(res1) != 0:
                regex_sum1 = re.compile("overall summary.+?qualification", flags=re.DOTALL)
                regex_sum2 = re.compile("overall summary.+?project description", flags=re.DOTALL)
                temp = re.findall(regex_sum1, data)
                for val in temp:
                    res2.append(skill_clean(val))

                if len(res2)==0:
                    temp = re.findall(regex_sum2, data)
                    for val in temp:
                        res2.append(skill_clean(val))

            else:
                regex_skills1 = re.compile(skl[0] + ".+?qualification", flags=re.DOTALL)
                regex_skills2 = re.compile(skl[0] + ".+?project description", flags=re.DOTALL)
                regex_sum1 = re.compile("overall summary.+?" + skl[0], flags=re.DOTALL)
                if "qualification" in data: 
                    regex_sum2 = re.compile("overall summary.+?qualification", flags=re.DOTALL)
                else: 
                    regex_sum2 = re.compile("overall summary.+?project description", flags=re.DOTALL)
                temp = re.findall(regex_skills1, data)
                for val in temp:
                    res1.append(skill_clean(val))
                    #print(res1)

                if len(res1)==0:
                    temp = re.findall(regex_skills2, data)
                    for val in temp:
                        res1.append(skill_clean(val))
                    #print(res1)

                temp = re.findall(regex_sum2, data)
                temp1 = re.findall(regex_sum1, data)
                if len(temp[0])<len(temp1[0]):
                    for val in temp:
                        res2.append(skill_clean(val))
                    #print(res2)

                if len(res2)==0:
                    temp = re.findall(regex_sum2, data)
                    for val in temp:
                        res2.append(skill_clean(val))
                    #print(res2)

            res1.extend(res2)
            skills = ""
            for val in res1:
                #print(val)
                skills = skills + " " + val
            res1.clear()
            #res1.append(skills)
            return skills

        else:
            regex_skills1 = re.compile(skl[0] + ".+?qualification", flags=re.DOTALL)
            regex_skills2 = re.compile(skl[0] + ".+?project description", flags=re.DOTALL)
            temp = re.findall(regex_skills1, data)
            for val in temp:
                res1.append(skill_clean(val))

            if len(res1)==0:
                temp = re.findall(regex_skills2, data)
                for val in temp:
                    res1.append(skill_clean(val))

            skills = ""
            for val in res1:
                skills = skills + " " + val
            res1.clear()
            #res1.append(skills)
            return skills

    elif "overall summary" in data:
        regex_sum1 = re.compile("overall summary.+?qualification", flags=re.DOTALL)
        regex_sum2 = re.compile("overall summary.+?project description", flags=re.DOTALL)
        temp = re.findall(regex_sum1, data)
        for val in temp:
            res2.append(skill_clean(val))

        if len(res2)==0:
            temp = re.findall(regex_sum2, data)
            for val in temp:
                res2.append(skill_clean(val))

        skills = ""
        for val in res2:
            skills = skills + " " + val
        res1.clear()
            #res1.append(skills)
        return skills

    else:
        return "data can not be parsed"

def listed_response(data):
    responsibility = []
    if "project description" in data:
        role = data.split("project description")
        role = role[1:]
        last = role[-1]
        if "areas of interest" in last:
            last = last.split("areas of interest")
            role[-1] = last[0]
        elif "goals" in last:
            last = last.split("goals")
            role[-1] = last[0]
            #print(role[-1])
        else:
            pass
        for val in role:
            responsibility.append(response_clean(val))
        #response = ""
        #for val in responsibility:
            #response = response + " " + val
        #print(response)
        #responsibility.clear()
        return responsibility

    elif "project" in data:
        role = data.split("project")
        role = role[1:]
        last = role[-1]
        if "areas of interest" in last:
            last = last.split("areas of interest")
            role[-1] = last[0]
        elif "goals" in last:
            last = last.split("goals")
            role[-1] = last[0]
        else:
            pass
        for val in role:
            responsibility.append(response_clean(val))

        #response = ""
        #for val in responsibility:
        #    response = response + " " + val

        #responsibility.clear()
        return responsibility

    else:
        return "data can not be parsed"

def response(data):
    responsibility = []
    if "project description" in data:
        role = data.split("project description")
        role = role[1:]
        last = role[-1]
        if "areas of interest" in last:
            last = last.split("areas of interest")
            role[-1] = last[0]
        elif "goals" in last:
            last = last.split("goals")
            role[-1] = last[0]
            #print(role[-1])
        else:
            pass
        for val in role:
            responsibility.append(response_clean(val))
        response = ""
        for val in responsibility:
            response = response + " " + val
        #print(response)
        responsibility.clear()
        return response

    elif "project" in data:
        role = data.split("project")
        role = role[1:]
        last = role[-1]
        if "areas of interest" in last:
            last = last.split("areas of interest")
            role[-1] = last[0]
        elif "goals" in last:
            last = last.split("goals")
            role[-1] = last[0]
        else:
            pass
        for val in role:
            responsibility.append(response_clean(val))

        response = ""
        for val in responsibility:
            response = response + " " + val

        responsibility.clear()
        return response

    else:
        return "data can not be parsed"

def clean_role(val):
    val = val.replace("\n", "")
    val = val.replace("\t", "")
    val = val.replace("role:", "")
    val = val.replace("name:", "")
    val = val.strip()
    role = val
    if len(role)>100:
        if "respons" in role:
            role = role.split("respons")
            role = role[0]
            return role
        else:
            chunks = [str[i:i+20] for i in range(0, len(role), 20)]
            role = chunks[0]
            return role
    return role

def clean_name(val):
    val = val.replace("\n", "")
    val = val.replace("name:", "")
    val = val.replace("role:", "")
    val = val.strip()
    name = val
    return name 

def role(data):
    regex_role1 = re.compile("role:.+?name:", flags=re.DOTALL)
    regex_role2 = re.compile("role:.+?\n\n", flags=re.DOTALL)
    temp = re.findall(regex_role1, data)
    #print(temp)
    if len(temp)!=0:
        #print(temp)
        return clean_role(temp[0])
    elif len(temp)==0:
        temp = re.findall(regex_role2, data)
        #print(temp)
        if len(temp)!=0:
            return clean_role(temp[0])
    else:
        return "data can not be parsed"

def name(data):
    regex_name2 = re.compile("name:.+?role:", flags=re.DOTALL)
    regex_name1 = re.compile("name:.+?\n", flags=re.DOTALL)
    temp = re.findall(regex_name1, data)
    temp1 = re.findall(regex_name2, data)
    #print(temp)
    #print(temp1)
    if temp:
        if temp1:
            if len(temp[0]) < len(temp1[0]):
                return clean_name(temp[0])
    if temp1:
        return clean_name(temp1[0])
    if temp:    
        return clean_name(temp[0])
    else:
        return "data can not be parsed"

def anotherExperience(content):
    years = re.findall(r'\d*\.*\d+[\s]?[+]?[\s]*years?', content)
    months = re.findall(r'\d*\.*\d+[\s]?[+]?[\s]*months?', content)
    if years != []:
        year_clean = [float(re.findall(r'\d*[.]?\d+', str(x))[0]) for x in years]
    else:
        year_clean = [0.0]
    if months != []:
        month_clean = [float(re.findall(r'\d*[.]?\d+', str(x))[0]) for x in months]
    else:
        month_clean = [0.0]
    total_years = max(year_clean) + max(month_clean)/12.0
    #return "{val:.2f}".format(val = total_years) + " year(s)"
    return total_years

def experience_returned(index_list, content):
    exp_lst = []
    for i in index_list:
        p = content[i-300:i+300]
        val = anotherExperience(p)
        if val:
            exp_lst.append(val)
    if exp_lst != []:
        total_years = max(exp_lst)
    else:
        total_years = 0.0
    return "{val:.2f}".format(val = total_years) + " year(s)"

def experience_extractor(resume_content):
    matcher = re.finditer('experience',resume_content)
    index_list = []
    for i in matcher:
        index_list.append(i.start())
    return experience_returned(index_list,resume_content)

def working_with_cv_content(Dataframe1):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf1 = TfidfVectorizer(analyzer='word',stop_words= 'english')
    vector1 = tf1.fit_transform(Dataframe1['combined_features']).toarray()  
    return vector1, tf1

def combined_features(row):
    return row['Skills']+" "+row['proj']

def remove_punct(doc):
  output = "".join([c for c in doc if c not in punct_list])
  return output
#canDF["Projects"].apply(lambda x: remove_punct(x))

def tokenized_words(doc):
  output = re.findall('\S+', doc)
  return output

#canDF["Projects"].apply(lambda x: tokenized_words(x))
def unstoppable_words(doc):
  output = [x for x in doc if x not in stop_words]
  return output

def generate_N_grams(text,ngram=2):
  #words=[word for word in text.split(" ") if word not in set(stop_words)]  
  #print("Sentence after removing stopwords:",words)
  text = [stemmer.stem(x) for x in text]
  temp=zip(*[text[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

def pipe_clean(doc):
  op = remove_punct(doc)
  op = tokenized_words(op.lower())
  op = unstoppable_words(op)
  if len(op) >= 2:
    nop = generate_N_grams(op, 2)
  else:
    nop = []
  # op = stem_words(op)
  op.extend(nop)
  return op, nop

if __name__ == "__main__":

    # PATHS
    PATH_BASE = os.path.dirname(os.path.abspath(__file__))
    #PATH_PARSE = os.path.join(PATH_BASE,'../output/AFour Profiles/model.pkl')
    PATH_PROCESSDF = os.path.join(PATH_BASE,'../../profiles/output/AFour Profiles/processDF.pkl')
    PATH_HRDF = os.path.join(PATH_BASE,'../../profiles/output/AFour Profiles/hrDF.xlsx')
    PATH_CVS = os.path.join(PATH_BASE,'../../profiles/input/AFour Profiles')
    PATH_FILE_CORRUPT = os.path.join(PATH_BASE,'../../profiles/output/AFour Profiles/unsupportedCVs')
    PATH_ARCHIVE = os.path.join(PATH_BASE,'../../profiles/archive/AFour Profiles')

    #################### author @Fahad ##################
    nltk.download('stopwords')
    sw_nltk = stopwords.words('english')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    directory = PATH_CVS
    #parsed = parser.from_file('../input/AFour Profile  - Tejas Mane.pdf')
    #data = parsed["content"] 
    lst = []
    #Collections_test = defaultdict(lambda: list)

    print('Reading Train Data')
    for filename in os.listdir(directory):
        fileUnsupported = False
        if filename.startswith('~$') or filename.startswith('$') or filename.startswith(".~"):
            print(" ----------- ")
            #print("---- RESUME START ------")
            print(filename)
            print("Ignored and moved..")
            fileUnsupported = True
        # elif not fileSupported(filename):
        #     print(" ----------- ")
        #     print(filename)
        #     print("Ignored and moved..")
        #     fileUnsupported = True
        else:
            try:
                #print(filename)
                data = convertAndParse(filename, directory+"/"+filename)
            except Exception as e: # flag unsupported formats
                fileUnsupported = True
                print(" -----ERROR------ ", e)
                print(filename)
                print("Ignored and moved..")
            if not fileUnsupported:
                data = data.lower()
                data = data.replace("professional summary", "overall summary")
                #data = data.replace("skill set summary", "skillset summary")
                #data = data.replace("skills summary", "skillset summary")
                #data = data.replace("skill summary", "skillset summary")
                #data = data.replace("skill-set summary", "skillset summary")
                data = data.replace("role :", "role:")
                data = data.replace("name :", "name:")
                #temp = open(path+folder+"/negative.review", 'r').read() # Read the file
                #temp = re.findall(regex_review, temp) # Get reviews
                #print("Reading",len(temp),"Negative reviews from",folder)
                #print(temp)
                
                record = []

                record.append(filename)
                record.append(name(data))
                record.append(role(data))
                record.append(skills_data(data))
                record.append(response(data))
                record.append(listed_response(data))
                record.append(experience_extractor(data))
                lst.append(record)
        if fileUnsupported:
            if filename.startswith(".~"):
                pass
            else:
                shutil.move(os.path.join(PATH_CVS, filename), os.path.join(PATH_FILE_CORRUPT, filename))
    #Collections_test[0].append(list)
    #print(Collections)

    new_df = pd.DataFrame(columns=['File', 'Name', 'Role', 'Skills', 'proj', 'Projects', 'Experience' ]
    , data=lst)
    new_df["combined_features"] = new_df.apply(combined_features, axis =1)

    vector_skills1, tf1 = working_with_cv_content(new_df)
    dtv= np.array(vector_skills1).tolist()
    rows = len(dtv)
    df2 = pd.DataFrame({"col":dtv, 'b':range(rows)})
    new_df['tfid_vec'] = df2['col'].values
    new_df.drop(columns=["combined_features", "proj"], axis=1, inplace=True)
    #pickle.dump(new_df, open(PATH_PARSE, 'wb'))

    ################ INITIAL PARSING DONE HERE ################

    ################ DATA FOR ALGORITHM AND MODEL SCORE ############
    ############### author @Siddhesh Girase ###################

    # load candidate docs from monthly pickle file for further processing
    # with open(PATH_PARSE, "rb") as f:
    #     canDF = pickle.load(f)
    # canDF = pd.DataFrame(object)
    canDF = new_df
    headers = ["File", "Name", "Role", "Experience", "tfid_vec"]
    processDF = canDF[headers].copy()         # to account for skill frequency in each project

    # Tokenization Functions

    #nltk.download('stopwords')

    # will be removed in future pre-processing steps
    except_list = ["#"]
    punct_list = [x for x in string.punctuation if x not in except_list]

    #print(punct_list)
    stop_words = nltk.corpus.stopwords.words('english')

    # for pre-processing pipeline
    stemmer = SnowballStemmer("english")

    # this anonymous function is used for adding skills and repeat project wise occurences for context.
    def f(x):
        temp = []
        for item in x:
            item = pipe_clean(item)[0]
            temp.extend(set(item))
        return temp

    processDF["Resume Doc"] = canDF["Skills"].apply(lambda x: pipe_clean(x)[0]) + canDF["Projects"].apply(lambda x: f(x))
    #processDF["Ex Check"] = processDF["Experience"]
    processDF["Experience"] = processDF["Experience"].apply(lambda x: re.findall(r'\d*[.]?\d+', str(x)))
    processDF["Experience"] = [float(x[0]) if len(x)>=1 else 0.0 for x in processDF["Experience"] ]

    # Add local file URLs
    def f(x):
        if x == 'AFour Profile - Sudhir Padalkar.docx':
            url = "https://afourtechpune-my.sharepoint.com/:w:/r/personal/admin_afourtech_com/Documents/From%20G%20Suite%20Drive/AFourCloudShare/ACS_HR/ACS_HR%20data/AFour%20profiles/AFour%20Profile%20-%20Sudhir%20Padalkar.docx?d=wffa836be4f7f4346a90c67344976c6b3&csf=1&web=1&e=JOSdJW"
        elif x == 'AFour Profile - Roshan Wakode.odt':
            url = "https://afourtechpune-my.sharepoint.com/:w:/r/personal/admin_afourtech_com/Documents/From%20G%20Suite%20Drive/AFourCloudShare/ACS_HR/ACS_HR%20data/AFour%20profiles/AFour%20Profile%20-%20Roshan%20Wakode.odt?d=w995f0f1c406d474792e731e93042342e&csf=1&web=1&e=MQwNw1"
        else:
            url = os.path.join(PATH_CVS[1:], x)
        return url
    processDF["URLs"] = processDF["File"].apply(f)

    #re.findall(r'\d*.\d+', processDF["Experience"][1])
    #processDF["Resume Doc"][0]
    temp = processDF == "data can not be parsed"
    hrDF = processDF[temp[temp.any(axis=1)] != None].dropna().copy()
    processDF = processDF.where(processDF != "data can not be parsed").dropna(how='any')
    processDF = processDF.sort_values(by=["Experience"], ascending=False).drop_duplicates(subset=["Name"])
    processDF.sort_index(inplace=True)
    
    # TEST
    #print(processDF["data can not be parsed"])
    #print(new_df.iloc[414, :])
    #print(hrDF)
    #print(processDF[["Name", "Ex Check", "Experience"]].to_string())
    #print(processDF[["URLs"]][0:1])

    if os.path.exists(PATH_PROCESSDF):
        # ct stores current time
        ct = time.strftime("%Y-%m-%d %H")
        #print("current time:-", ct)
        shutil.move(PATH_PROCESSDF, os.path.join(PATH_ARCHIVE, str(ct) + "_processDF.pkl"))
        #os.rename(my_source, my_dest)
    if os.path.exists(PATH_HRDF):
        # ct stores current time
        ct = time.strftime("%Y-%m-%d %H")
        #print("current time:-", ct)
        shutil.move(PATH_HRDF, os.path.join(PATH_ARCHIVE, str(ct) + "_hrDF.xlsx"))
        #os.rename(my_source, my_dest)
    pickle.dump(processDF, open(PATH_PROCESSDF, 'wb'))
    hrDF.to_excel(PATH_HRDF)
    ################### DUMPED FILE TO BE USED FOR SCORING THE RESUMES ######################

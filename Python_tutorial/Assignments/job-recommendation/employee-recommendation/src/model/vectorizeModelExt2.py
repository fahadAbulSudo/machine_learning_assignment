import os
import pickle
import re
import shutil
import string
import time

import nltk
import numpy as np
import pandas as pd
from convertAndParse1 import convertAndParse
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

import gc
gc.set_threshold(0)

def getName(filename):
  file = filename.split(".")
  file = " ".join([x for x in file[:-1]])
  #print(file)
  plist = ["_"]
  plist.extend(punct_list)
  clean = "".join([c if c not in plist and not c.isdigit() else " " for c in file])
  #print(clean)
  nameList = tokenized_words(clean.lower())
  #print(nameList)
  name = ""
  for item in nameList:
    if len(item) >= 3 and not\
    re.match("resume|profile|vitae|curriculum|exp|pdf|doc|docx|years?|months?", item):
      if name == "":
        name = item
      else:
        name += " " + item
  if name == "":
    name = "Missing Field!"
  return name

def fileSupported(filename):
  supportFormats = [".docx", ".doc", ".dot", ".dotx", ".odt", ".pdf", ".txt"]
  supported = False
  for extension in supportFormats:
    if filename.endswith(extension) or filename.endswith(extension.upper()):
      supported = True
  return supported

def getRole(data):
  data = data.lower()
  data = remove_punct(data)
  swPostList = re.findall("(\S+)\s*(\S+)\s*(engineer|architect|scientist)", data)
  postList = []
  for item in swPostList:
    role = " ".join([remove_punct(x) for x in item if len(remove_punct(x)) >= 4])
    postList.append(role)
  #swPostList = re.findall(r"([a-zA-Z]+\s+){engineer|architect|scientist}()(\s+[a-zA-Z]+){2}", data)
  postList = ", ".join([x for x in postList[:1]])
  # if postList == "":
  #   postList = "Missing Field!"
  return postList

def getSkills(data, width=50):
  data = pipe_clean(data)[0]
  limitcontext = []
  for wcount in range(0, len(data), width):
    limitcontext.extend(list(set(data[wcount:wcount+width])))
    #limitcontext = [list(x) for x in limitcontext]
  #print(limitcontext)
  #print(len(limitcontext))
  return limitcontext

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
      total_years = "Missing Field!"
  return total_years
  #return "{val:.2f}".format(val = total_years) + " year(s)"

def experience_extractor(resume_content, filename):
  if re.match(r"\[\d+y_\d+m\]", filename):
    yearsNaukri = re.match(r"\[\d+y_\d+m\]", filename).string[1:-2]
    yearsNaukri = yearsNaukri.split("y_")
    experience = float(yearsNaukri[0]) + float(yearsNaukri[1])/12.
  else:
    matcher = re.finditer('experienc',resume_content)
    index_list = []
    noMatch = True
    for i in matcher:
      noMatch = False
      index_list.append(i.start())
      experience = experience_returned(index_list,resume_content)
    if noMatch:
      experience = "Missing Field!"
  #print(experience)
  return experience

def working_with_cv_content(Dataframe1):
  from sklearn.feature_extraction.text import TfidfVectorizer
  Dataframe1["doc"] = Dataframe1['Resume Doc'].apply(lambda x: " ".join([word for word in x]))
  tf1 = TfidfVectorizer(analyzer='word',stop_words= 'english')
  vector1 = tf1.fit_transform(Dataframe1['doc']).toarray()  
  return vector1, tf1

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

    # files and base path
    PATH_BASE = os.path.dirname(os.path.abspath(__file__))
    #PATH_PARSE = os.path.join(PATH_BASE,'../output/parseExt.pkl')
    PATH_PROCESSDF = os.path.join(PATH_BASE,'../../profiles/output/External Profiles/processDFExt.pkl')
    PATH_HRDF = os.path.join(PATH_BASE,'../../profiles/output/External Profiles/hrDFExt.xlsx')

    # directories
    PATH_EXT_OP = os.path.join(PATH_BASE,'../../profiles/output/External Profiles')
    if not os.path.exists(PATH_EXT_OP):
      os.mkdir(PATH_EXT_OP)
    PATH_MINI = os.path.join(PATH_BASE,'../../profiles/output/External Profiles/tempMini')
    if not os.path.exists(PATH_MINI):
      os.mkdir(PATH_MINI)
    PATH_MINI_DF = os.path.join(PATH_BASE,'../../profiles/output/External Profiles/tempMini/processDFExt.pkl')
    PATH_HRDF_MINI_DF = os.path.join(PATH_BASE,'../../profiles/output/External Profiles/tempMini/hrDFExt.csv')
    PATH_CVS = os.path.join(PATH_BASE,'../../profiles/input/External Profiles')
    if not os.path.exists(PATH_CVS):
      os.mkdir(PATH_CVS)
    PATH_FILE_CORRUPT = os.path.join(PATH_BASE,'../../profiles/output/External Profiles/unsupportedCVs')
    if not os.path.exists(PATH_FILE_CORRUPT):
      os.mkdir(PATH_FILE_CORRUPT)
    PATH_ARCHIVE = os.path.join(PATH_BASE,'../../profiles/archive/External Profiles')
    if not os.path.exists(PATH_ARCHIVE):
      os.mkdir(PATH_ARCHIVE)

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
    
    # Tokenization Functions
    #nltk.download('stopwords')
    # will be removed in future pre-processing steps
    except_list = ["#"]
    punct_list = [x for x in string.punctuation if x not in except_list]
    punct_list.extend(['‘', '➢', '\uf02a'])
    #print(punct_list)
    stop_words = nltk.corpus.stopwords.words('english')
    # for pre-processing pipeline
    stemmer = SnowballStemmer("english")

    print('Reading Train Data')

    # BATCH INFORMATION
    batch_size = 20
    _, count = 0, 0

    for filename in os.listdir(directory):
      if count == 0:
         totalData = []
      fileUnsupported = False

      if filename.startswith('~$') or filename.startswith('$'):
        print(" ----------- ")
        #print("---- RESUME START ------")
        print(filename)
        print("Ignored and moved..")
        fileUnsupported = True
      # elif not fileSupported(filename):
      #   print(" ----------- ")
      #   print(filename)
      #   print("Ignored and moved..")
      #   fileUnsupported = True
      else:
        print(_, filename)
        _ += 1
        # if _ <= 625:
        #    continue
        
        try: # read as many formats as possible
          data = convertAndParse(filename, directory+"/"+filename)
          count += 1

        except Exception as e: # flag unsupported formats
          fileUnsupported = True
          print(" ----ERROR----- ", e)
          print(filename)
          print("Ignored and moved..")
        
        if not fileUnsupported:
          record = []
          record.append(filename)
          record.append(getName(filename))
          if data != None or data != '' or data != []: # pdf not in image format
            record.append(getRole(data))
            record.append(experience_extractor(data.lower(), filename))
            record.append(getSkills(data))
          else: # when pdf in image format
            fileUnsupported = True
            record.append("Missing Field!")
            record.append("Missing Field!")
            record.append("Missing Field!")
          
          totalData.append(record)
          del(data)
          del(record) 

      if fileUnsupported:
        shutil.move(os.path.join(PATH_CVS, filename), os.path.join(PATH_FILE_CORRUPT, filename))
      
      if count < batch_size:
         continue
      else:
        batch_num = str(int(_/batch_size))
        count = 0
        print("wait processing last batch...")
        headers = ["File", "Name", "Role", "Experience", "Resume Doc"]
        processDF = pd.DataFrame(totalData, columns=headers)
        del(totalData)
        #pickle.dump(totalData, open(PATH_MINI_DF+"/"+batch_num+".pkl", 'wb'))
        vector_skills1, tf1 = working_with_cv_content(processDF)
        dtv= np.array(vector_skills1).tolist()
        rows = len(dtv)
        df2 = pd.DataFrame({"col":dtv, 'b':range(rows)})
        processDF['tfid_vec'] = df2['col'].values
        del(vector_skills1)
        del(tf1)
        processDF["Experience"] = processDF["Experience"].apply(lambda x: re.findall(r'\d*[.]?\d+', str(x)))
        processDF["Experience"] = [float(x[0]) if len(x)>=1 else 0.0 for x in processDF["Experience"] ]
        processDF.drop('doc', inplace=True, axis=1)
        temp = processDF == "Missing Field!"
        hrDF = processDF[temp[temp.any(axis=1)] != None].dropna().copy()
        processDF = processDF.sort_values(by=["Experience"], ascending=False).drop_duplicates(subset=["File"])
        processDF.sort_index(inplace=True)
        if os.path.exists(PATH_MINI_DF):
          df = pickle.load(open(PATH_MINI_DF, 'rb'))
          DF = processDF.append(df, ignore_index=True)
          print(PATH_MINI_DF) 
          del df   
        else:
          DF = processDF.copy(deep=True)
        if os.path.exists(PATH_HRDF_MINI_DF):
          df1 = pd.read_csv(PATH_HRDF_MINI_DF)
          #df1 = pickle.load(open(PATH_HRDF_MINI_DF, 'rb'))
          hrDF = hrDF.append(df1, ignore_index=True)
          print(PATH_HRDF_MINI_DF) 
          del df1   
        '''else:
          DF = processDF.copy(deep=True)'''

        pickle.dump(DF, open(PATH_MINI_DF, 'wb'))
        hrDF.to_csv(PATH_HRDF_MINI_DF)
        del(processDF)
        del(hrDF)
        time.sleep(1)

    '''totalData = []
    for filename in os.listdir(PATH_MINI_DF):
      with open(PATH_MINI_DF+"/"+filename,'rb') as file:
        mini_data = pickle.load(file)
      #print(mini_data[0])
      totalData.extend(mini_data)
      os.remove(PATH_MINI_DF+"/"+filename)'''
  
    
    
    
    

    # Add local file URLs
    '''def f(x):
        if x == 'AFour Profile - Sudhir Padalkar.docx':
            url = "https://afourtechpune-my.sharepoint.com/:w:/r/personal/admin_afourtech_com/Documents/From%20G%20Suite%20Drive/AFourCloudShare/ACS_HR/ACS_HR%20data/AFour%20profiles/AFour%20Profile%20-%20Sudhir%20Padalkar.docx?d=wffa836be4f7f4346a90c67344976c6b3&csf=1&web=1&e=JOSdJW"
        elif x == 'AFour Profile - Roshan Wakode.odt':
            url = "https://afourtechpune-my.sharepoint.com/:w:/r/personal/admin_afourtech_com/Documents/From%20G%20Suite%20Drive/AFourCloudShare/ACS_HR/ACS_HR%20data/AFour%20profiles/AFour%20Profile%20-%20Roshan%20Wakode.odt?d=w995f0f1c406d474792e731e93042342e&csf=1&web=1&e=MQwNw1"
        else:
            url = os.path.join(PATH_CVS[1:], x)
        return url
    processDF["URLs"] = processDF["File"].apply(f)'''
    #re.findall(r'\d*.\d+', processDF["Experience"][1])
    #processDF["Resume Doc"][0]
    
    #processDF = processDF.where(processDF != "Missing Field!").dropna(how='any')

    

    #processDF.to_excel(PATH_BASE+"/test.xlsx")
    if os.path.exists(PATH_PROCESSDF):
      # ct stores current time
      ct = time.strftime("%Y-%m-%d %H")
      #print("current time:-", ct)
      shutil.move(PATH_PROCESSDF, os.path.join(PATH_ARCHIVE, str(ct) + "_processDFExt.pkl"))
      #os.rename(my_source, my_dest)
    shutil.move(PATH_MINI_DF, PATH_PROCESSDF)
    if os.path.exists(PATH_HRDF):
      # ct stores current time
      ct = time.strftime("%Y-%m-%d %H")
      #print("current time:-", ct)
      shutil.move(PATH_HRDF, os.path.join(PATH_ARCHIVE, str(ct) + "_hrDFExt.xlsx"))
      #os.rename(my_source, my_dest)
    shutil.move(PATH_HRDF_MINI_DF, PATH_HRDF)

    #print(processDF)
    

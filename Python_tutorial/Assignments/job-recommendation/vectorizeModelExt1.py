from tika import parser
import re
import pickle
import pandas as pd
import numpy as np
import nltk
import re
import os
import sys
import string
import shutil
import PyPDF2
import time
import boto3
import requests
import datetime
from datetime import datetime
import logging
import sys
import threading
from queue import Queue
from docx import Document
from io import BytesIO
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# from collections import defaultdict
from nltk.corpus import stopwords



global AWS_ACCESS_KEY
global AWS_SECRET_KEY
global AWS_REGION_NAME
global BUCKET
global URL
global DESTINATION
 

AWS_REGION_NAME = "ap-south-1"
BUCKET = "cv-filtering"

class AWSS3(object):
 
    __slots__ = ["BucketName", "client", "resource"]
 
    def __init__(self, BucketName = BUCKET):
        self.BucketName = BucketName
        self.client = boto3.client("s3",
                                   aws_access_key_id=AWS_ACCESS_KEY,
                                   aws_secret_access_key=AWS_SECRET_KEY,
                                   region_name=AWS_REGION_NAME)
        self.resource = boto3.resource('s3',
                                   aws_access_key_id=AWS_ACCESS_KEY,
                                   aws_secret_access_key=AWS_SECRET_KEY,
                                   region_name=AWS_REGION_NAME)

    def putFiles(self, Response=None, Key=None):
        """
        Put the File on S3
        :return: Bool
        """
        try:
            #client = boto3.client('s3')
            copy_source = {
            'Bucket': 'cv-filtering',
            'Key': Key
            }
            self.resource.meta.client.copy(copy_source, 'cv-filtering', '/output/External Profiles/unsupportedCVs/')
            return 'ok'
        except Exception as e:
            print("Error : {} ".format(e))
            return 'error'

    def ItemExists(self, Key):
        try:
            # get the Response for teh Current File
            response_new = self.client.get_object(Bucket=self.BucketName, Key=str(Key))
            return True
        except Exception as e:
            return False
 
    def getItem(self, Key):
        try:
            response_new = self.client.get_object(Bucket=self.BucketName, Key=str(Key))
            return response_new["Body"].read()
        except Exception as e:
            return False

    def operation(self, data=None, key=None):
 
        """
        This checks if Key is on S3 if it is return the data from s3
        else store on s3 and return it
        """
 
        flag = self.ItemExists(Key=key)
        if flag:
            data = self.getItem(Key=key)
            return data

    def getAllKeys(self ,Prefix=''):
        """
        :param Prefix: Prefix string
        :return: Keys List
        """
        # Paginator Objects
 
        paginator = self.client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.BucketName, Prefix=Prefix)
 
        tmp = []
 
        for page in pages:
            for obj in page['Contents']:
                tmp.append(obj['Key'])
 
        return tmp

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
  supportFormats = [".docx", ".doc", ".dot", ".dotx", ".odt", ".pdf", ".txt", ".rtf"]
  supportFormats1 = [".dotx", ".odt", ".pdf", ".txt"]
  supported = False
  for extension in supportFormats:
    if filename.endswith(extension) or filename.endswith(extension.upper()):
      ext = extension
      supported = True
      return supported, ext
  for extension in supportFormats1:
    if filename.endswith(extension) or filename.endswith(extension.upper()):
      ext = extension
      supported = True
      return supported, ext
  return supported, "null"

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

class AWSFileReader(object):
 
    """High level class responsible for serializing """
 
    def __init__(self, fileName):
        self.fileName  = fileName
        self.dataqueue = Queue()
 
    def get(self):
        try:
            supportFormats = [".docx", ".doc", ".dot", ".dotx", ".odt", ".pdf", ".txt", ".rtf"]
            _instance = AWSS3()
            record = []
            filelst = self.fileName.split("/")
            file = filelst[-1]
            if file.startswith('~$') or file.startswith('$'):
              print(" ----------- ")
              #print("---- RESUME START ------")
              print(file)
              print("Ignored and moved..")
              record.append(file)
              record.append("Missing Field!")
              record.append("Missing Field!")
              record.append("Missing Field!")
                #print(experience_extractor(data.lower()))
              record.append("Missing Field!")
              filesupported = True
            elif file.endswith((".docx", ".doc", ".dot", ".dotx", ".odt", ".pdf", ".txt", ".rtf")):
              print(" ----------- ")
              #print("---- RESUME START ------")
              print(file)
              file1 = _instance.getItem(Key=self.fileName)
              data = parser.from_buffer(BytesIO(file1))["content"]
              record.append(file)
              record.append(getName(file))
              if data != None: # pdf not in image format
                record.append(getRole(data))
                record.append(experience_extractor(data.lower(), file))
                #print(experience_extractor(data.lower()))
                record.append(getSkills(data))
                filesupported = True
              else:
                filesupported = False
                record.append("Missing Field!")
                record.append("Missing Field!")
                #print(experience_extractor(data.lower()))
                record.append("Missing Field!")
              del data
            else: 
              print(file)
              print("Ignored and moved..")
              record.append(file)
              record.append("Missing Field!")
              record.append("Missing Field!")
              record.append("Missing Field!")
                #print(experience_extractor(data.lower()))
              record.append("Missing Field!")
              filesupported = False

            if fileSupported == False:
              _instance.putFiles(Key=file)
            self.dataqueue.put(record)
            return record
        except Exception as e:
            self.dataqueue.put('error')
            return 'error'

if __name__ == "__main__":

    # PATHS
    PATH_BASE = os.getcwd()
    print(PATH_BASE)
    #PATH_PARSE = os.path.join(PATH_BASE,'../output/parseExt.pkl')
    PATH_PROCESSDF = PATH_BASE.replace("\\src\\model","") 
    PATH_PROCESSDF = PATH_PROCESSDF + '\\profiles\\output\\External Profiles\\processDFExt.pkl'
    PATH_HRDF = PATH_BASE.replace("\\src\\model","")
    PATH_HRDF = PATH_HRDF + '\\profiles\\output\\External Profiles\\hrDFExt.xlsx'
    PATH_CVS = PATH_BASE.replace("\\src\\model","")
    PATH_CVS = PATH_CVS + '\\profiles\\input\\External Profiles'
    PATH_FILE_CORRUPT = PATH_BASE.replace("\\src\\model","")
    PATH_FILE_CORRUPT = PATH_FILE_CORRUPT + '\\profiles\\output\\External Profiles\\unsupportedCVs'
    PATH_ARCHIVE = PATH_BASE.replace("\\src\\model","")
    PATH_ARCHIVE = PATH_ARCHIVE + '\\profiles\\archive\\External Profiles'

    #nltk.download('stopwords')
    sw_nltk = stopwords.words('english')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

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
    directory = PATH_CVS
    totalData = []
    _instance = AWSS3()
    Keys = _instance.getAllKeys(Prefix='input/External Profiles/')
 
    MAIN_BATCH_SIZE = 100
 
    MainThreads = []
    Instances = []
    for key in Keys:
 
        _AWSFileReader = AWSFileReader(fileName=key)
        t = threading.Thread(target=_AWSFileReader.get)
        MainThreads.append(t)
        Instances.append(_AWSFileReader)
    l = MainThreads
    batch_size = MAIN_BATCH_SIZE
 
    """for each batch """
    for i in range(0, len(l), batch_size):
 
        print("working on Main Batch : {} ".format(i))
 
        flush_variable = []                             # Fush Variable
        # [thread1, thread 2, ......]
        data = l[i:i+batch_size]                        # get the Group of Thread Objects
 
        # [xaspp0012, 211321 ........]
        json_records = Instances[i:i+batch_size]        # get the instance batch
 
        for thread in data : thread.start()                                             # Start the Batch Threads
        for thread in data : thread.join()                                              # Wait for batch Thread to complete
        for instance in json_records:flush_variable.append(instance.dataqueue.get())    # get the data from instance of class and queue object
        print()
        # your Batch data
        totalData.extend(flush_variable)
    '''for filename in os.listdir(directory):
        #filesupported, extension = fileSupported(filename)
        if filename.startswith('~$') or filename.startswith('$'):
            print(" ----------- ")
            #print("---- RESUME START ------")
            print(filename)
            print("Ignored and moved..")
            filesupported = False
        elif filename.endswith(".docx"):# or filename.endswith(".doc"):
            lst = []
            print(" -----------doc ")
            print(filename)
            doc = Document(os.path.join(directory, filename))
            for para in doc.paragraphs:
                line = para.text
                lst.append(line)
            data = ''.join(lst)
            #totalData.append(data)
            record = []
            #print(getName(filename))
            #print(getRole(data))
            record.append(filename)
            record.append(getName(filename))
            if data != None: # pdf not in image format
                record.append(getRole(data))
                record.append(experience_extractor(data.lower(), filename))
                #print(experience_extractor(data.lower()))
                record.append(getSkills(data))
                filesupported = True
            else: # when pdf in image format
                record.append("Missing Field!")
                record.append("Missing Field!")
                #print(experience_extractor(data.lower()))
                record.append("Missing Field!")
                filesupported = False
            del data
            del doc
            del para
            totalData.append(record)
            del record
            #totalData.append(record)
        elif filename.endswith(".pdf"):
            lst = []
            print(" -----------pdf ")
            print(filename)
            pdfFileObj = open(os.path.join(directory, filename), 'rb')
      
            # creating a pdf reader object
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            
            # printing number of pages in pdf file
            try: 
                for i in range(len(pdfReader.pages)):
                    pageObj = pdfReader.pages[i]
                    page = pageObj.extract_text()
                    lst.append(page)
                data = ''.join(lst)
                # closing the pdf file object
                pdfFileObj.close()
                #totalData.append(data)
                record = []
                #print(getName(filename))
                #print(getRole(data))
                record.append(filename)
                record.append(getName(filename))
                if data != None: # pdf not in image format
                    record.append(getRole(data))
                    record.append(experience_extractor(data.lower(), filename))
                    #print(experience_extractor(data.lower()))
                    record.append(getSkills(data))
                    filesupported = True
                else: # when pdf in image format
                    filesupported = False
                    record.append("Missing Field!")
                    record.append("Missing Field!")
                    #print(experience_extractor(data.lower()))
                    record.append("Missing Field!")
            except KeyError:
                record.append(filename)
                record.append("Missed")
                record.append("Missed")
                record.append("Missed")
                    #print(experience_extractor(data.lower()))
                record.append("Missed")
                filesupported = False
            del data
            del lst
            totalData.append(record)
            del record
                #print(".tika")
                #print(filename)
                #parsed = parser.from_file(os.path.join(directory, filename))
                #print(parsed)
                #data = parsed["content"]
                #totalData.append(data)
                #record = []
                #print(getName(filename))
                #print(getRole(data))
                #record.append(filename)
                #record.append(getName(filename))
                #if data != None: # pdf not in image format
                #    record.append(getRole(data))
                #    record.append(experience_extractor(data.lower(), filename))
                #    #print(experience_extractor(data.lower()))
                #    record.append(getSkills(data))
		            
        else:
            filesupported = False
            print(".tika")
            print(filename)
            parsed = parser.from_file(os.path.join(directory, filename))
            #print(parsed)
            data = parsed["content"]
            #totalData.append(data)
            record = []
            #print(getName(filename))
            #print(getRole(data))
            record.append(filename)
            record.append(getName(filename))
            if data != None: # pdf not in image format
                record.append(getRole(data))
                record.append(experience_extractor(data.lower(), filename))
                #print(experience_extractor(data.lower()))
                record.append(getSkills(data))
            else: # when pdf in image format
                filesupported = False
                record.append("Missing Field!")
                record.append("Missing Field!")
                #print(experience_extractor(data.lower()))
                record.append("Missing Field!")
            totalData.append(record)
        
        if filesupported == False:
            import shutil
            shutil.move(os.path.join(PATH_CVS, filename), os.path.join(PATH_FILE_CORRUPT, filename)) '''
  
    headers = ["File", "Name", "Role", "Experience", "Resume Doc"]
    processDF = pd.DataFrame(totalData, columns=headers)
    
    vector_skills1, tf1 = working_with_cv_content(processDF)
    dtv= np.array(vector_skills1).tolist()
    rows = len(dtv)
    df2 = pd.DataFrame({"col":dtv, 'b':range(rows)})
    processDF['tfid_vec'] = df2['col'].values

    processDF["Experience"] = processDF["Experience"].apply(lambda x: re.findall(r'\d*[.]?\d+', str(x)))
    processDF["Experience"] = [float(x[0]) if len(x)>=1 else 0.0 for x in processDF["Experience"] ]
    processDF.drop('doc', inplace=True, axis=1)

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
    temp = processDF == "Missing Field!"
    hrDF = processDF[temp[temp.any(axis=1)] != None].dropna().copy()
    #processDF = processDF.where(processDF != "Missing Field!").dropna(how='any')
    processDF = processDF.sort_values(by=["Experience"], ascending=False).drop_duplicates(subset=["File"])
    processDF.sort_index(inplace=True)
    #processDF.to_excel(PATH_BASE+"/test.xlsx")

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

    #print(processDF) 

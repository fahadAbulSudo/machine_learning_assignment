import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import string
from math import exp
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn import preprocessing
PATH_nltk = os.path.dirname(os.path.abspath(__file__))
PATH_nltk = PATH_nltk + '/nltk_data'
#PATH_nltk = os.path.join(PATH_BASE,'/nltk_data')

def customSkillScorer(userReq: dict, candidateDoc: list, candidateEx: float= -1.0, candidateRole: str = None
  , skillSigRatio: list = None, missPenaltyRatio: list = None):
  '''
  skills: our query doc for skills currently in dict format
  candidateDoc: our candidates' tokenized document list
  skillSigRatio: a list containiing weights of each category of skills for scoring
  missPenaltyRatio: a list containiing penalty power weights of each category of skills for scoring
  '''
  # Settable weightage parameters
  # must have skills
  wt1, pow1, countu1, countf1 = 8.0, 2, 0, 0
  # 7 * (x/N)^2
  # similar skills
  wt2, pow2, countu2, countf2 = 0.0, 1, 0, 0
  # good to have
  wt3, pow3, countu3, countf3 = 3.0, 1, 0, 0
  # 3 * x/N

  # default max experience adder and weight for experience
  addEx, wtx = 5, 3

  ## 2 / (1 + e**-0.1(x - N))

  def skillFreqBonus(x, N, bonus = 2, linear = 0.1):
    temp = bonus/(1 + exp(-linear*(x-N)))
    if temp > 1:
      extra = temp - 1
      temp = 1 + extra/2.0
    return temp

  # if scoring weightage needs to change
  if skillSigRatio != None:
    wt1, wt2, wt3 = skillSigRatio[0], skillSigRatio[1], skillSigRatio[3]
  if missPenaltyRatio != None:
    pow1, pow2, pow3 = skillSigRatio[0], skillSigRatio[1], skillSigRatio[3]

  # load query data
  skills = userReq["skills"]
  userEx = userReq["experience"]
  if candidateRole != None:
    desig = userReq["designation"]
  mustHave = list(set(skills['mustHave']))
  similar = list(set(skills['similar']))
  goodToHave = list(set(skills['goodToHave']))
  #minEx, maxEx = canEx["min"], canEx["max"] # make userEx
  minEx = userEx["min"]
  
  # if max ex not given use 5 as default years of plateau
  # if maxEx == None:
  #   maxEx = minEx + addEx

  doc_unique = list(set(candidateDoc))
  doc_freq = list(candidateDoc)         # account multiple occurences as well
  
  for item in doc_unique:
    if (item in mustHave):
      countu1 += 1
    if (item in similar):
      countu2 += 1
    if (item in goodToHave):
      countu3 += 1

  for item in doc_freq:
    if (item in mustHave):
      countf1 += 1
    if (item in similar):
      countf2 += 1
    if (item in goodToHave):
      countf3 += 1

  # Unique words score
  N1, N2, N3 = len(mustHave), 0, len(goodToHave)
  score1 = wt1 * (countu1/N1)**pow1
  #score2 = wt2 * (countu2/N2)**pow2
  score2 = 0
  score3 = wt3 * (countu3/N3)**pow3

  # multiply it by bonus frequency score to account for hands-on experience
  score1 *= skillFreqBonus(countf1, N1)
  #score2 *= skillFreqBonus(countf2, N2)
  score3 *= skillFreqBonus(countf3, N3)

  # till 2 - 4 years make plateau. then 1.5 times
  # Experience score - a gaussian curve

  if minEx <= 2:
    maxEx = 4
  else:
    maxEx = minEx*1.5
  spread = (maxEx - minEx)*addEx
  center = maxEx
  
  if candidateEx == -1.0:
    scoreEx = 1.0
  elif candidateEx < minEx or candidateEx > maxEx:
    scoreEx = wtx * exp(-(candidateEx-center)**2/spread)
  else:
    scoreEx = wtx
  
  overall = (score1 + score2 + score3 + scoreEx)
  # penalize by 40% for experience missing resumes
  #return overall/(wt1+wt2+wt3), score1/wt1, score2/wt2, score3/wt3
  return overall*100/(wt1+wt2+wt3+wtx), score1*100/wt1, score3*100/wt3, scoreEx*100/wtx



#PATH_nltk_data = '/home/fahad/Documents/cvfilter/app/nltk_data'
# Tokenization Functions
#print(PATH_nltk)
nltk.data.path.append(PATH_nltk)
#nltk.download('stopwords')

# will be removed in future pre-processing steps
except_list = ["#"]
punct_list = [x for x in string.punctuation if x not in except_list]

#print(punct_list)
stop_words = nltk.corpus.stopwords.words('english')

# pre processing pipeline

stemmer = SnowballStemmer("english")
stemmer.stem('lovingly')

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

def recommendTopMatch(userMust, userGood, minEx, processDF):
  def f(x):
    temp = pipe_clean(x)
    if temp[1] != []:
      temp = temp[1]
    else:
      temp = temp[0]
    if temp == []:
      temp = ['']
    return temp
  
  # ERROR HANDLING
  # if userMust == ['']:
  #   userMust = ['somerandomword']
  # if userGood == ['']:
  #   userGood = ['somerandomword']
  tuserMust = np.concatenate([x for x in [f(x) for x in userMust] if x != []])
  tuserGood = np.concatenate([x for x in [f(x) for x in userGood] if x != []])

  userReq = {
      "skills": {"mustHave": tuserMust,
                "similar": ["Python", "Octave", "NLP"],
                "goodToHave": tuserGood
      },
      "experience": {"min": minEx},
      #"designation": {"min": "Trainee", "max": "Sr. SDE"}
  }

  shortHeads = ["Name", "Role", "Experience", "File", "id"]
  scoreHeads = ["Compatibility %", "Must Have Skills %", "Good to Have %", "Experience Match %"]
  recommendDF = processDF[shortHeads].copy()

  # just to score each candidate entry
  def f(x):
      return customSkillScorer(userReq, x[0], x[1])
  recommendDF[scoreHeads] = list(processDF[["Resume Doc", "Experience"]].apply(f, axis=1))
  # x = recommendDF[scoreHeads].values #returns a numpy array
  # min_max_scaler = preprocessing.MinMaxScaler((0,100))
  # x_scaled = min_max_scaler.fit_transform(x)
  # recommendDF[scoreHeads] = x_scaled

  sortSettings = scoreHeads
  #sortSettings = ["Must Have Skills %", "Compatibility %", "Good to Have %", "Experience Match %"]
  #sortSettings = ["Over and Above %"]
  #recommendDF["id"] = recommendDF.index
  recommendDF = recommendDF.sort_values(by=sortSettings, ascending=False)
  
  #recommendDF.index = np.arange(1, len(recommendDF) + 1)#for streamlit
  
  recommendDF.Name = recommendDF.Name.str.title()
  recommendDF.Role = recommendDF.Role.str.upper()
  return recommendDF


def similarCandidates(searchDF, recommendDF, ids, nSims = 10):
  if len(ids) > 10:
    id = ids[0:10]
  else:
    id = ids
  # process vector dimensions
  count = len(id)
  a = 10 // count   #this is the value having minimum number of employee per recommendation
  #print(a)
  b = 10 % count   #THis is extra if remaining
  d = {}  #Forming dictionary having vectors as well as how many employee per similar recomendations can be done 
  e = 0  #To check how many employee can be given extra similar candidates
# iterating through the elements of list
  for i in id:
    l = []
    if e < b:
      l.append(a + 2)
      tlist = searchDF['skills2vec'][searchDF.index==i].values
      l.append(tlist)
      d[i] = l
    else:
      l.append(a + 1)
      tlist = searchDF['skills2vec'][searchDF.index==i].values
      l.append(tlist)
      d[i] = l
    e += 1
  import pandas as pd
  similar_df = pd.DataFrame()#columns=['File', 'Name', 'Role', 'Experience', 'Resume Doc', 'skills2vec','results'])
  def simi(vec,skills2vec):
    from scipy import spatial
    result = 1 - spatial.distance.cosine(vec, skills2vec) #Doing cosine similarity
    return result
    
  def score(df,vec1,n,dff):
    #vec1 = df[df['File']==filename]['skills2vec'].values[0]
    #print(n)
    df['results'] = df.apply(lambda x: simi(vec1,x.skills2vec), axis=1)#Id column
    df1 = df.sort_values(by = 'results', ascending = False) 
    #print(df1)
    if dff.shape[0] == 0:
      df2 = df1.iloc[0:n]
      #In this condition block checking if similar datframe having some value if not directly append value to it 
      dff = pd.concat([dff, df2], axis=0)
      #print(dff)
      return dff
    df2 = df1.iloc[0:20]
    lst2 = df2.values.tolist()
    lst3 = dff.values.tolist()
    #print(len(lst2[0]))
    #print(lst3)
    c = 0
    o = 0
    #print(len(lst3))
    
    while c < n:
      j = 0
      #This loop is major importance it is checking first if same IDs are selected or not if yes then taking higher result into consideration
      for row in lst3:
        #print(row[0])
        if row[6] == lst2[o][6]:
          
          #print(lst2[o][0])
          if row[-1] < lst2[o][-1]:
            row[-1] = lst2[o][-1]
        else:
          j += 1
      if j == len(lst3):
        lst3.append(lst2[o])
        o += 1
        c += 1
        #print(o)
        #print(c)
      else:
        o += 1
      
    '''for row in lst3:
      print(row[0])'''
    headers = ["File", "Name", "Role", "Experience", "Resume Doc", "skills2vec", "id", "results"]
    rslt_df = pd.DataFrame(lst3, columns=headers) #add d column
    rslt_df = rslt_df.sort_values(by = 'results', ascending = False)
    return rslt_df
  for key,value in d.items():
    #print(value[0])
    #print(key)
    list_1 = value[1].tolist()
    similar_df = score(searchDF,list_1[0],value[0],similar_df)
    #similar_df = similar_df.append(m)
  similar_df = similar_df.iloc[count:]
  similar_df.drop('skills2vec', inplace=True, axis=1)
  similar_df.drop('Resume Doc', inplace=True, axis=1)
  similar_df.drop('results', inplace=True, axis=1)
  similar_df.Name = similar_df.Name.str.title()
  similar_df.Role = similar_df.Role.str.upper()
  recommendDF.drop('Name', inplace=True, axis=1)
  recommendDF.drop('Role', inplace=True, axis=1)
  recommendDF.drop('Experience', inplace=True, axis=1)
  recommendDF.drop('File', inplace=True, axis=1)
  #print(similar_df.shape)
  #print(recommendDF.shape)
  int_df = pd.merge(similar_df, recommendDF, how ='inner', on =['id'])
  #print(int_df.columns)
  return int_df


'''def similarCandidates(searchDF, recDF, id, nSims = 10):
  # process vector dimensions
  tlist = searchDF["tfid_vec"][searchDF.index.isin(id)].values
  vecs = []
  for i in tlist:
    vecs.append(i)
  #print(np.shape(vecs))
  meanVec = np.mean(np.array(vecs).T, axis=1)           # representation vector
  temp = searchDF.drop(index = id)                      # remove previous matches
  
  def f(x):
    x = np.array(x).reshape(meanVec.shape)
    return cosine_similarity([meanVec, x])[0][1]
  temp["similarity score"] = temp["tfid_vec"].apply(f) 
  cscores = temp.sort_values(by=["similarity score"], ascending=False).head(nSims)["similarity score"]
  simID = cscores.index
  simDF = recDF[recDF.index.isin(simID)].copy()
  simDF["Similarity %"] = cscores.apply(lambda x: x*100)

  simDF.Name = simDF.Name.str.title()
  simDF.Role = simDF.Role.str.upper()
  return simDF.sort_values(by=["Similarity %"], ascending=False)'''
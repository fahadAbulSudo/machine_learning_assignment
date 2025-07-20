import ast
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stems(text):
  """Applying stemming:
        input:
            text: content
  """  
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)  

def recommend(item,df,similarity):
    """Recommend function
    Input:
        movie: name of the movie
        df   : dataframe
        similarity: cosine similarity 
    """
    index = df[df['title'] == item].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(df.iloc[i[0]].title)      
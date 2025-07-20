from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn import preprocessing
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import random
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


def iqr(df, val):
    Q1 = df[val].quantile(0.05)
    Q3 = df[val].quantile(0.95)
    IQR = Q3 - Q1
    return df[(df[val]>= Q1 - 1.5*IQR) & (df[val] <= Q3 + 1.5*IQR)]

def process_check(data):
    columns = data.columns
    for i in columns:
        if data.dtypes[i] is object:
            print("Cannot process object data")
        

    ## if skew is 0.5 and -0.5 i.e its normal distribution and use scaler else normalise
    scale_list = []
    normalise_list = []
    data_skew = data.skew()

    for i in columns:
        if data_skew[i] >= -0.5 and data_skew[i] <= 0.5:
            scale_list.append(i)
        else:
            normalise_list.append(i)

    return scale_list, normalise_list

def process_normal(df, normalise_list):
    features = df[normalise_list]
    normal = preprocessing.MinMaxScaler().fit(features.values)
    features = normal.transform(features.values)
    return features, normal

def process_standard(df, scale_list):
    features = df[scale_list]
    standard = preprocessing.StandardScaler().fit(features.values)
    features = standard.transform(features.values)
    return features, standard

def process_labelencode(df, val, model = False):
    le = preprocessing.LabelEncoder()
    le.fit(df[val])
    if model:
	    return le.transform(df[val]), le
    else:
        return le.transform(df[val])

def ra(pre):   
    test_list1 = [1,2]
    test_list2 = [1, 4, 13, 9, 20]
    for i in range(0,len(pre)):
        a = random.choice(test_list1)
        b = random.choice(test_list2)
        if a == 1:
            pre[i] = pre[i] - b
        else:
            pre[i] = pre[i] + b

    return pre

def corr_mat(df):
    plt.figure(figsize = (10,5))
    sns.heatmap(df.corr(),annot = True , cmap = 'coolwarm' );

def co_mat(df, val):
    corr_mat = df.corr()
    plt.figure(figsize = (10,5))
    corr_mat[val].sort_values(ascending = False).plot(kind = 'bar'); 

def RandomizedSCV(model,parameters,verbose, n_jobs, n_iter, X_train, y_train):
    random_search = RandomizedSearchCV(model,param_distributions=parameters,verbose=verbose, n_jobs=n_jobs, n_iter=n_iter)
    random_result = random_search.fit(X_train, y_train)
    return random_result




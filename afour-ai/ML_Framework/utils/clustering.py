from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

# Plot Cluster Profile based on given parameters
# df : data frame, x_para_name: Column name for x-axis, y_para_name: Column name for y-axis, cluster_col_name: Cluster column name,
# plot_type: type of plot to be used to plot data, pal= Color pallete
def plot_cluster_profile(df, x_para_name, y_para_name, cluster_col_name="Clusters", plot_type="scatterplot", pal=None):
  if pal == None:
    pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
  
  #check if column name specified exists or not
  if x_para_name not in df:
    return f"{x_para_name} column name doesn't exist"
  elif y_para_name not in df:
    return f"{y_para_name} column name doesn't exist"
  elif cluster_col_name not in df:
    return f"{cluster_col_name} column name doesn't exist"
  

  if plot_type == "scatterplot":
    pl = sns.scatterplot(data = df,x=df[x_para_name], y=df[y_para_name], hue=df[cluster_col_name], palette= pal)
    pl.set_title(f"Cluster's Profile Based On {x_para_name} And {y_para_name}")
    plt.legend()
    plt.show()
  elif plot_type == "boxplot":
    print("For box plot the clusters are taken as x-axis and y_para_name is taken as y-axis")
    plt.figure()
    pl=sns.boxenplot(x=df[cluster_col_name], y=df[y_para_name], palette=pal)
    plt.show()
  elif plot_type == "joinplot":
    plt.figure()
    sns.jointplot(x=df[x_para_name], y=df[y_para_name], hue=df[cluster_col_name], kind="kde", palette=pal)
    plt.show()
  else:
    print(f"{plot_type} not supported. Only [boxplot,joinplot,scatterplot] are supported!")


# Plot cluster distribution given the data frame and the cluster column name
def plot_cluster_distribution(df, cluster_col_name = "Clusters"):
  pl = sns.countplot(x=df[cluster_col_name])
  pl.set_title("Distribution Of The Clusters")
  plt.show()



# Determine the best number of clusters given the dataframe and scoring method.
def ideal_number_of_clusters(df, method = "Elbow"):
  #check if df has high dimenionsality if yes reduce by using PCA
  if method == "Elbow":
    print('Elbow Method to determine the number of clusters to be formed:')
    Elbow_M = KElbowVisualizer(KMeans(), k=10)
    Elbow_M.fit(df)
    Elbow_M.show()
  elif method == "Silhouette":
    print('Silhouette Score to determine the number of clusters to be formed:')
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    max = 0
    ideal_k = 0
    for num_clusters in range_n_clusters:
      # intialise kmeans
      kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
      kmeans.fit(df)
      
      cluster_labels = kmeans.labels_
      
      # silhouette score
      silhouette_avg = silhouette_score(df, cluster_labels)
      if silhouette_avg > max:
        max = silhouette_avg
        ideal_k = num_clusters
      #print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    print(f"Best silhouette score obtained for cluster : {ideal_k}")
  else:
    print(f"{method} not supported!")








# Importing libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import seaborn as sns
# Importing dataset

dataset=pd.read_csv(r"C:\Users\Hemant Ojha\OneDrive\Desktop\Finlatics1\MLResearch\MLResearch\Facebook Dataset\Facebook_Marketplace_data.csv")

dataset.info()
# Q1. How does the time of upload (`status_published`)  affects the `num_reaction`?

# convert the feature 'status_published' into datetime format

dataset['status_published']=pd.to_datetime(dataset['status_published'])

# Extract hour from 'status_published'
dataset['hour_published'] = dataset['status_published'].dt.hour

# Group by 'hour_published' and calculate the average of 'num_reaction'
hourly_reactions = dataset.groupby('hour_published')['num_reactions'].mean()
print(hourly_reactions)

# Graphical visualization of time of upload  affects the `num_reaction`
plt.figure(figsize=(10, 6))
plt.plot(hourly_reactions.index, hourly_reactions.values, marker='o', linestyle='-', color='blue')
plt.title('Average Number of Reactions vs. Hour of Upload')
plt.xlabel('Hour of Upload')
plt.ylabel('Average Number of Reactions')
plt.grid(True)
plt.xticks(range(0, 24))  # Setting x-axis ticks to represent hours
plt.show()



'''
Q2. Is there a correlation between the number of reactions (num_reactions) and other engagement metrics 
such as comments (num_comments) and shares (num_shares)? If so, what is the strength and direction of this 
correlation?
'''

# Correlation between num_reactions, num_comments and num_shares


# creating variable
numeric_data=dataset[['num_reactions','num_comments','num_shares']].select_dtypes(include=['int64'])
# Creating correlation matrix of num_reactions and num_comments
corr_matrix=numeric_data.corr()
print(corr_matrix)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='PuBuGn',fmt='.2f')
plt.title('correlation between num_reactions and num_comments')
plt.show()

# so correlation between num_reactions and num_comments is 0.15 therefore it is weak correlation but it 
# positive hence it's direction is positive direction that is when one variable increses then other also increase


# Now correlation between num_reactions and num_shares is 0.25 therefore it is weak correlation but it 
# positive hence it's direction is positive direction that is when one variable increses then other also increase




'''
Q3.	Use the columns status_type, num_reactions, num_comments, num_shares, num_likes, num_loves, num_wows, 
num_hahas, num_sads, and num_angrys to train a K-Means clustering model on the Facebook Live Sellers dataset.
'''


x=dataset[['status_type','num_reactions','num_comments','num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys']]

dataset.isnull().sum()

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)

dataset.head(10)

# Using the elbow method to find the optimal number of clusters

wcss=[]

for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss,marker='o')  # range(1,11) for 10 clusters used in loop
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the KMeans model on the dataset

kmeans=KMeans(n_clusters=9,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)

# Dimensionality Reduction using PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# visualizing the clusters

# Plotting clusters
plt.figure(figsize=(10, 8))
plt.scatter(x_pca[y_kmeans == 0, 0], x_pca[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(x_pca[y_kmeans == 1, 0], x_pca[y_kmeans == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(x_pca[y_kmeans == 2, 0], x_pca[y_kmeans == 2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(x_pca[y_kmeans == 3, 0], x_pca[y_kmeans == 3, 1], s=50, c='yellow', label='Cluster 4')
plt.scatter(x_pca[y_kmeans == 4, 0], x_pca[y_kmeans == 4, 1], s=50, c='orange', label='Cluster 5')
plt.scatter(x_pca[y_kmeans == 5, 0], x_pca[y_kmeans == 5, 1], s=50, c='gray', label='Cluster 6')
plt.scatter(x_pca[y_kmeans == 6, 0], x_pca[y_kmeans == 6, 1], s=50, c='cyan', label='Cluster 7')
plt.scatter(x_pca[y_kmeans == 7, 0], x_pca[y_kmeans == 7, 1], s=50, c='brown', label='Cluster 8')
plt.scatter(x_pca[y_kmeans == 8, 0], x_pca[y_kmeans == 8, 1], s=50, c='magenta', label='Cluster 9')

# Plotting centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=100, c='black', label='Centroids')

plt.title('Facebook Live Sellers Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()



# Q4. Use the elbow method to find the optimum number of clusters.
# Ans:- from above code in  3rd question the optimal number of cluster is 9



# Q5. What is the count of different types of posts in the dataset?

dataset.columns

# status_type feature about types of posts

count_posts=dataset['status_type'].value_counts()
print(count_posts)

# Answer
'''
photo     4288
video     2334
status     365
link        63
'''

# Q6. What is the average value of num_reaction, num_comments, num_shares for each post type?

# Group by 'status_type' and calculate the average of 'num_reaction', 'num_comments', and 'num_shares'

average_value=dataset.groupby('status_type')[['num_reactions','num_comments','num_shares']].mean()
print(average_value)

# Answer
'''
             num_reactions  num_comments  num_shares
status_type                                         
link            370.142857      5.698413    4.396825
photo           181.290345     15.993470    2.553871
status          438.783562     36.238356    2.558904
video           283.409597    642.478149  115.679949
'''











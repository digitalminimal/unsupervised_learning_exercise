
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



# Load dataframe
df = pd.read_csv('https://raw.githubusercontent.com/lucko515/clustering-python/master/Customer%20in%20Mall%20clusterng/Mall_Customers.csv')


def exploredataframe(data):
    print("\n===Data Types :=== \n" + str(data.dtypes))
    print("=======================")
    print("\n\n\n Data Describe :\n" + str(data.describe()))
    print("=======================")
    # print("\nData is null  \n"+str(data.isnull()))
    print("\\n\n\n Data is null SUM \n" +str( data.isnull().sum())) #looks at the values inside.
    print("\n Data is null count \n" +str(data.count())) # looks at columns avaiable, doesnt care if i
    missing_values = data.isnull().sum().sort_values(ascending=False)
    percent_of_missing_values = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data= pd.concat([missing_values,percent_of_missing_values], axis=1, keys=['Total','Percent'])
    print("=======================")
    print("\\n\n\n Mising Data " + str(missing_data.head(20)))
    print("=======================")



    print(set(df['Genre']))
obj_df = df.select_dtypes(include=['object']).copy()
newfeature=pd.get_dummies(obj_df, columns=["Genre"])
newfeature


df=df.drop(['Genre'], axis=1)
print(df)

result = pd.concat([df, newfeature], axis=1)
result

# result=result.drop(['Genre'], axis=1)
result=result.drop(['CustomerID'], axis=1)

bins = [0, 25, 35, 45, 55, 65, 100]
result['binned'] = pd.cut(result['Age'], bins).astype(str)
binned=result.drop(['Age'], axis=1)
print(binned)

binned['binned'].unique()


print(binned)

# set figure size
plt.rcParams["figure.figsize"] = (12,8)

# plot clusters
plt.scatter(binned.iloc[: , 0],
            binned.iloc[: , 1],
            c='black',
            marker='o')
plt.xlabel("Annual Income (k$)"),
plt.ylabel("Spending Score (1-100)")
plt.grid()
plt.show()




km = KMeans(n_clusters=5) # how many clusters we expected  # how many initial runs
 #always produce the same sequence of k-means 



# fit and predict
binned=result.drop(['binned'], axis=1)
print(binned)
km.fit(df)
# y_km = km.fit_predict(binned)

#heigh and weight 
#fit finds the weight of the formula ( fit (coeficient) times values feature + (fit ) bias ( is approximate ) )




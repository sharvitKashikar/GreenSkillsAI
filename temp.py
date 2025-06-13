import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv(r'C:\Users\priya\OneDrive\Desktop\greenAI\day-8\KNN\Social_Network_Ads.csv')

df.head()
df['Purchased'].value_counts()


gender = pd.get_dummies(df['Gender'],drop_first=True)

df = pd.concat([df, Gender], axis=1)
df.drop(['Gender'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

scaled_features
df_feat = pd.DataFrame(scaled_features, columns=X.columns)
df_feat.head()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']


knn = KNeighborsClassifier
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
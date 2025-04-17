import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("creditcard.csv")
df.head()

df.shape

df.columns

print(df["Class"].value_counts(normalize=True) * 100)

df.isnull().sum()

df.duplicated().sum()

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

x= df.drop(['Class'], axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


input_layer = Input(shape=(x_train.shape[1],))

encoded = Dense(14, activation ='relu')(input_layer)
encoded = Dense(7, activation = 'relu')(encoded)

bottleneck = Dense(3, activation ='relu')(encoded)

decoded = Dense(7, activation='relu')(bottleneck)
decoded = Dense(14, activation='relu')(decoded)
decoded = Dense(x_train.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, epochs=10, batch_size=32, shuffle=True, validation_data=(x_test, x_test))


x_pred = autoencoder.predict(x_test)

mse = np.mean(np.power(x_test - x_pred, 2), axis=1)

threshold = np.percentile(mse, 95)
fraud_predictions = (mse> threshold).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, fraud_predictions))


kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(x)

# Check fraud distribution per cluster
print(df.groupby("Cluster")["Class"].value_counts(normalize=True))


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=["Class"]))

pca = PCA(n_components=0.95)  
df_pca = pca.fit_transform(df_scaled)


plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.show()


precision = precision_score(y_test, fraud_predictions)
recall = recall_score(y_test, fraud_predictions)
f1 = f1_score(y_test, fraud_predictions)

print("📊 Evaluation Metrics")
print(f"Precision Score: {precision:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Full report
print("\nClassification Report:")
print(classification_report(y_test, fraud_predictions))



import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("creditcard.csv")
df

# Features only (exclude 'Class')
X = df.drop(columns=['Class'])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
iso_model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
y_pred_iso = iso_model.fit_predict(X_scaled)

# Map predictions: -1 → fraud (1), 1 → normal (0)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

# Evaluation
print("🧊 Isolation Forest Results:")
print(confusion_matrix(df['Class'], y_pred_iso))
print(classification_report(df['Class'], y_pred_iso))




# LOF is unsupervised and fit-predict in one step
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
y_pred_lof = lof_model.fit_predict(X_scaled)

# Map predictions
y_pred_lof = [1 if x == -1 else 0 for x in y_pred_lof]

# Evaluation
print("🔍 Local Outlier Factor Results:")
print(confusion_matrix(df['Class'], y_pred_lof))
print(classification_report(df['Class'], y_pred_lof))




# Use only normal transactions (Class = 0) to train Autoencoder
normal_data = df[df['Class'] == 0].drop(columns=['Class'])
fraud_data = df[df['Class'] == 1].drop(columns=['Class'])

# Scale
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_data)
all_scaled = scaler.transform(df.drop(columns=['Class']))


input_dim = normal_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu', activity_regularizer=regularizers.l1(1e-5))(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(normal_scaled, normal_scaled,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_split=0.2,
                verbose=1)


reconstructions = autoencoder.predict(all_scaled)
mse = np.mean(np.power(all_scaled - reconstructions, 2), axis=1)

# Threshold based on normal data's MSE
threshold = np.percentile(mse, 99.9)
df['autoencoder_score'] = mse
df['autoencoder_pred'] = (mse > threshold).astype(int)

# Evaluation
print("⚙️ Autoencoder Results:")
print(confusion_matrix(df['Class'], df['autoencoder_pred']))
print(classification_report(df['Class'], df['autoencoder_pred']))


# Helper for metric calculation
def evaluate_model(y_true, y_pred, model_name):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"Model": model_name, "Precision": p, "Recall": r, "F1-Score": f1}

# Results
results = []
results.append(evaluate_model(df['Class'], y_pred_iso, "Isolation Forest"))
results.append(evaluate_model(df['Class'], y_pred_lof, "Local Outlier Factor"))
results.append(evaluate_model(df['Class'], df['autoencoder_pred'], "Autoencoder"))

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

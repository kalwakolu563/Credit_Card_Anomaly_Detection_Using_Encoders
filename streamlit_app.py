import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Title
st.title("🚨 Credit Card Fraud Detection - Autoencoder Model")

# Upload CSV
uploaded_file = st.file_uploader("creditcard.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Preprocess
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
    x = df.drop(['Class'], axis=1)
    y = df['Class']

    # Load or Re-train Model
    input_dim = x.shape[1]

    # Define Autoencoder again (simplified for demo)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(14, activation='relu')(input_layer)
    encoded = Dense(7, activation='relu')(encoded)
    bottleneck = Dense(3, activation='relu')(encoded)
    decoded = Dense(7, activation='relu')(bottleneck)
    decoded = Dense(14, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train model
    autoencoder.fit(x, x, epochs=5, batch_size=32, verbose=0)

    # Predict
    x_pred = autoencoder.predict(x)
    mse = np.mean(np.power(x - x_pred, 2), axis=1)
    threshold = np.percentile(mse, 95)

    y_pred = (mse > threshold).astype(int)

    # Show metrics
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.subheader("📈 Model Evaluation Metrics")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

    # Show Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

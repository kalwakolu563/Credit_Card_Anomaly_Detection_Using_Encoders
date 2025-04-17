import pandas as pd
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("creditcard.csv")
df

df['Time'] = pd.to_datetime(df['Time'], unit='s')
df['Time'] = df['Time'] - df['Time'].min()
df['Time'] = pd.to_timedelta(df['Time'])

# Create a proper datetime index
df['Timestamp'] = pd.Timestamp("2020-01-01") + df['Time']
df.set_index('Timestamp', inplace=True)

# Transaction Frequency (per hour/day)

# Minute frequency
df['minute'] = df.index.floor('min')  # 'T' = minute
minute_freq = df.groupby('minute').size()
df['minute_freq'] = df['minute'].map(minute_freq)

# Hourly frequency
df['hour'] = df.index.floor('h')  # 'H' = hour
hourly_freq = df.groupby('hour').size()
df['hourly_freq'] = df['hour'].map(hourly_freq)

# Daily frequency
df['day'] = df.index.floor('D')  # 'D' = day
daily_freq = df.groupby('day').size()
df['daily_freq'] = df['day'].map(daily_freq)

# View result
df[['Amount', 'minute_freq', 'hourly_freq', 'daily_freq']].head()

# Moving Average of Spending (short-term trends)

# Rolling window of 1 minute
df['rolling_amount_1min'] = df['Amount'].rolling('1min').mean()

# Rolling window of 60 minutes
df['rolling_amount_1h'] = df['Amount'].rolling('60min').mean()
df['rolling_amount_count_1h'] = df['Amount'].rolling('60min').count()
# Rolling window of 180 minutes (3 hours)
df['rolling_amount_3h'] = df['Amount'].rolling('180min').mean()

# View the result
df[['Amount', 'rolling_amount_1min','rolling_amount_count_1h', 'rolling_amount_1h', 'rolling_amount_3h']].head()


# Ratio of High-Value Transactions

# Define high-value
df['high_value_flag'] = df['Amount'] > 200

# Ratio over last 60 minutes
df['high_value_ratio_1h'] = df['high_value_flag'].rolling('60min').mean()

df[['Amount', 'high_value_flag', 'high_value_ratio_1h']].head()

# ✅ Feature Summary Table

features = ['Amount', 'hourly_freq', 'daily_freq', 
            'rolling_amount_1h', 'rolling_amount_3h',
            'high_value_flag', 'high_value_ratio_1h']

df[features].describe()

# Statistical Anomaly Detection

# Z-Score Method

# Calculate z-scores
df['z_score'] = zscore(df['Amount'])

# Set threshold for anomaly
threshold = 4
df['z_score_outlier'] = (df['z_score'].abs() > threshold).astype(int)

# Show counts
print(df['z_score_outlier'].value_counts())

# Plot the Outliers

plt.figure(figsize=(12, 6))

# Plot normal points (z_score_outlier = 0)
sns.scatterplot(x=df.index, y=df['Amount'], hue=df['z_score_outlier'],
                palette={0: 'blue', 1: 'red'}, alpha=0.6)

plt.title('Credit Card Transactions with Z-score Outliers')
plt.xlabel('Transaction Index')
plt.ylabel('Transaction Amount')
plt.legend(title='Outlier', labels=['Normal', 'Outlier'])
plt.yscale('log')  # Optional: if amount range is large

plt.tight_layout()
plt.show()

# IQR Method

# Calculate IQR
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Flag anomalies
df['iqr_outlier'] = ((df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)).astype(int)

# Show counts
print(df['iqr_outlier'].value_counts())

# Plotting the IQR Outliers:

# Set up the figure size
plt.figure(figsize=(12, 6))

# Plot all transaction amounts
sns.scatterplot(x=df.index, y=df['Amount'], color='blue', label='Normal Transactions', alpha=0.6)

# Highlight the outliers with a scatter plot (red crosses for outliers)
sns.scatterplot(x=df.index[df['iqr_outlier'] == 1], 
                y=df['Amount'][df['iqr_outlier'] == 1], 
                color='red', label='Outliers', s=100, marker='x')

# Add titles and labels
plt.title('Scatter Plot of Transaction Amounts with IQR Outliers')
plt.xlabel('Transaction Index')
plt.ylabel('Transaction Amount')

# Display the legend and plot
plt.legend(title='Outlier Status')
plt.tight_layout()
plt.show()

# Compare with Actual Fraud (Class) Using Confusion Matrix



# Z-score evaluation
print("🔍 Z-Score Method vs Actual Fraud:")
print(confusion_matrix(df['Class'], df['z_score_outlier']))
print(classification_report(df['Class'], df['z_score_outlier']))

# IQR evaluation
print("🔍 IQR Method vs Actual Fraud:")
print(confusion_matrix(df['Class'], df['iqr_outlier']))
print(classification_report(df['Class'], df['iqr_outlier']))


# ✅ Visualize Outliers

plt.figure(figsize=(10, 4))
sns.scatterplot(data=df, x='Amount', y='z_score', hue='z_score_outlier', palette={0: 'blue', 1: 'red'})
plt.title("Z-Score Based Outliers")
plt.axhline(y=threshold, color='green', linestyle='--')
plt.axhline(y=-threshold, color='green', linestyle='--')
plt.show()



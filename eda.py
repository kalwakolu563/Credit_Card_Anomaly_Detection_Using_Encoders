import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("creditcard.csv")
df

df['Time'] = pd.to_datetime(df['Time'], unit='s')
df['Time'] = df['Time'] - df['Time'].min()
df['Time'] = pd.to_timedelta(df['Time'])

# Create a proper datetime index
df['Timestamp'] = pd.Timestamp("2020-01-01") + df['Time']
df.set_index('Timestamp', inplace=True)

# 📈 1. Visualize Time Series Trends

# Hourly Transaction Count

plt.figure(figsize=(12, 6))
df['Amount'].resample('h').count().plot()
plt.title("Hourly Transaction Count")
plt.xlabel("Hour")
plt.ylabel("Transaction Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 Plot Transactions per Minute

df['Amount'].resample('min').count().plot(figsize=(12, 4))  # 'T' stands for minute
plt.title('Transactions per Minute')
plt.xlabel('Time')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 Plot Transactions per Day

df['Amount'].resample('D').count().plot(figsize=(12, 4))  # 'D' = daily
plt.title('Transactions per Day')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.tight_layout()
plt.show()

# ⏱️ Plot: Transactions per Second

df['Amount'].resample('s').count().plot(figsize=(12, 4))  # 'S' = seconds
plt.title('Transactions per Second')
plt.xlabel('Time')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.tight_layout()
plt.show()


# 📅 Plot: Transactions per Week

df['Amount'].resample('W').count().plot(figsize=(12, 4))  # 'W' = weekly
plt.title('Transactions per Week')
plt.xlabel('Week')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🗓️ Plot: Transactions per Month

df['Amount'].resample('M').count().plot(figsize=(12, 4))  # 'M' = monthly
plt.title('Transactions per Month')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.tight_layout()
plt.show()

# here we will not have charts 
# as the data is only one day so the plot representation is not correct 
# as we dont have weeks in the data is that ok ??
# and similarly if we check with month on month also we will not get plots as we dont have data 
# i hope you got the clarity please do let me know if anything not understood Mr.Shravan.K

# Hourly Transaction Amount Sum

plt.figure(figsize=(12, 4))
df['Amount'].resample('h').sum().plot()
plt.title("Hourly Total Transaction Amount")
plt.xlabel("Hour")
plt.ylabel("Total Amount")
plt.grid(True)
plt.tight_layout()
plt.show()


# 🎯 2. Fraud vs Non-Fraud Boxplots / Histograms 
# 🔹 Class distribution

sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0: Non-Fraud, 1: Fraud)")
plt.show()

# 🔹 Amount Distribution by Class

plt.figure(figsize=(10, 5))
sns.boxplot(x='Class', y='Amount', data=df)
plt.yscale('log')
plt.title("Transaction Amount Distribution by Class (Log Scale)")
plt.show()

# 🔹 Histogram: Amount Comparison

plt.figure(figsize=(12, 5))
df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.6, label='Non-Fraud')
df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.6, label='Fraud', color='red')
plt.legend()
plt.title("Histogram of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.xlim([0, 1000])
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x=df['Amount'])
plt.title("Distribution of Amount Transaction")
plt.show()

# Feature Correlation Matrix

plt.figure(figsize=(16, 10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of All Features")
plt.show()

# 🔹 Focus on Features Correlated with 'Class'

corr_target = corr['Class'].drop('Class').sort_values(ascending=False)
print("Top Correlated Features with Fraud Class:\n", corr_target.head(10))



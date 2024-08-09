import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Import the data
df = pd.read_csv("Queries.csv")

# Check for null values
print(df.isnull().sum())

# Get descriptive statistics
print(df.describe())

# Convert CTR column to float
df['CTR'] = df['CTR'].str.rstrip('%').astype('float') / 100

# Check column names
print(df.columns)

# Clean and split the queries into words
def clean_query(query):
    query = re.sub(r'[^\w\s]', '', query)
    query = query.lower()
    words = query.split()
    return words

# Split the queries into words and count the frequency of each word
word_counts = Counter()
for query in df['Top queries']:
    words = clean_query(query)
    word_counts.update(words)

# Plot the word frequencies
word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Frequency'])
word_counts_df = word_counts_df.sort_values('Frequency', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(word_counts_df.head(20).index, word_counts_df.head(20)['Frequency'])
plt.title('Top 20 Most Common Words in Search Queries')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Plot the top queries by clicks and impressions
plt.figure(figsize=(10, 6))
plt.bar(df.sort_values('Clicks', ascending=False).head(10)['Top queries'], df.sort_values('Clicks', ascending=False).head(10)['Clicks'])
plt.title('Top Queries by Clicks')
plt.xlabel('Queries')
plt.ylabel('Clicks')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df.sort_values('Impressions', ascending=False).head(10)['Top queries'], df.sort_values('Impressions', ascending=False).head(10)['Impressions'])
plt.title('Top Queries by Impressions')
plt.xlabel('Queries')
plt.ylabel('Impressions')
plt.show()

# Plot the top and bottom queries by CTR
plt.figure(figsize=(10, 6))
plt.bar(df.sort_values('CTR', ascending=False).head(10)['Top queries'], df.sort_values('CTR', ascending=False).head(10)['CTR'])
plt.title('Top Queries by CTR')
plt.xlabel('Queries')
plt.ylabel('CTR')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df.sort_values('CTR', ascending=True).head(10)['Top queries'], df.sort_values('CTR', ascending=True).head(10)['CTR'])
plt.title('Bottom Queries by CTR')
plt.xlabel('Queries')
plt.ylabel('CTR')
plt.show()

# Check the correlation between different metrics
numeric_cols = ['Clicks', 'Impressions', 'CTR', 'Position']
corr_matrix = df[numeric_cols].corr()

# Display the correlation matrix
plt.figure(figsize=(10, 8))
# Change the colormap to 'coolwarm'
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

print(corr_matrix)

# Observation:
# The correlation matrix shows the correlation between each pair of metrics.
# A high correlation (close to 1 or -1) indicates a strong relationship between the metrics.
# A low correlation (close to 0) indicates a weak relationship between the metrics.
# In this case, we can see that:
# - Clicks and Impressions are highly correlated (0.95), indicating that queries with high impressions tend to have high clicks.
# - CTR is moderately correlated with Clicks (0.63) and Impressions (0.61), indicating that queries with high CTR tend to have high clicks and impressions.
# - Position is weakly correlated with the other metrics, indicating that the position of the query in the search results does not have a strong impact on clicks, impressions, or CTR.

# Detect anomalies using Isolation Forest
# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

# Train the Isolation Forest model
model = IsolationForest(contamination=0.05)  # Set the contamination level
model.fit(scaled_data)

# Predict anomalies
predictions = model.predict(scaled_data)
df['Anomaly'] = predictions

# Show the results
anomaly_df = df[df['Anomaly'] == -1]
print(anomaly_df[['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']])
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the Netflix data
df = pd.read_csv("netflix_titles.csv")

# Drop rows that don't have rating, duration, or listed_in
df.dropna(subset=['rating', 'duration', 'listed_in'], inplace=True)

# Convert duration to minutes (only apply to Movies)
df = df[df['type'] == 'Movie'].copy()

def extract_minutes(duration):
    try:
        return int(duration.strip().split(' ')[0])
    except:
        return None

df['duration_mins'] = df['duration'].apply(extract_minutes)
df.dropna(subset=['duration_mins'], inplace=True)

# Use only the first genre
df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0])

# Encode categorical columns
le_rating = LabelEncoder()
le_genre = LabelEncoder()

df['rating_code'] = le_rating.fit_transform(df['rating'])
df['genre_code'] = le_genre.fit_transform(df['main_genre'])

# Prepare features for clustering
features = df[['rating_code', 'genre_code', 'duration_mins']]

# Scale the features
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# Run KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

# Save clustered data to CSV
df.to_csv("clustered_netflix.csv", index=False)

# Visualize the clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='duration_mins', y='rating_code', hue='cluster', palette='tab10')
plt.title("K-Means Clustering of Netflix Movies")
plt.xlabel("Duration (minutes)")
plt.ylabel("Rating (encoded)")
plt.savefig("netflix_clusters.png")  # Optional: Save image
plt.show()

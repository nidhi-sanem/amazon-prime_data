# Step 1: Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load Dataset
df = pd.read_csv('amazon_prime_titles.csv')

# Step 3: Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(subset=['type', 'title'], inplace=True)
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['duration'] = df['duration'].fillna('0')
df['duration_int'] = df['duration'].str.extract('(\d+)').astype(float)
df['duration_type'] = df['duration'].str.extract('([a-zA-Z]+)').fillna('')

# Step 4: EDA
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='type')
plt.title("Distribution of Content Types")
plt.show()

plt.figure(figsize=(10,4))
df['rating'].value_counts().plot(kind='bar')
plt.title("Rating Distribution")
plt.show()

plt.figure(figsize=(12,6))
df['release_year'].value_counts().head(20).plot(kind='bar')
plt.title("Content by Year of Release (Top 20)")
plt.show()

# WordCloud for genres
plt.figure(figsize=(12, 6))
text = ' '.join(str(g) for g in df['listed_in'].dropna())
wordcloud = WordCloud(background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Genres WordCloud')
plt.show()

# Step 5: Simple Content-Based Filtering (based on listed_in)
def recommend_by_genre(genre_keyword):
    mask = df['listed_in'].str.contains(genre_keyword, case=False, na=False)
    return df[mask][['title', 'listed_in', 'description']].head(10)

# Example usage:
print("Recommendations for Comedy:\n", recommend_by_genre("Comedy"))

# Step 6: Predictive Modeling
# Let's predict the type (Movie or TV Show) based on other features
model_df = df[['type', 'duration_int', 'release_year']]
model_df.dropna(inplace=True)
model_df['type'] = model_df['type'].map({'Movie': 0, 'TV Show': 1})

X = model_df[['duration_int', 'release_year']]
y = model_df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 🎬 **Streaming Wars: Analyzing OTT Content & Building a Recommendation System**  

**Skills:** Data Analytics, Machine Learning, Content Recommendation, Python, Pandas, Seaborn  

---

## 🚀 **Project Overview**  
The explosion of **OTT streaming platforms** has led to an overwhelming amount of content. Users often struggle to decide **what to watch next**. This project tackles this problem by analyzing **content from major streaming platforms** and building a **personalized movie recommendation system**.  

✅ **Analyzed Streaming Platforms:**  
- **Amazon Prime Video**  
- **Apple TV+**  
- **Disney+**  
- **HBO Max**  
- **Netflix**  
- **Paramount+**  
- **Hulu**  

📌 **Datasets from Kaggle:**  
- [Amazon Prime Video](https://www.kaggle.com/datasets/dgoenrique/amazon-prime-movies-and-tv-shows)  
- [Apple TV+](https://www.kaggle.com/datasets/dgoenrique/apple-tv-movies-and-tv-shows)  
- [Disney+](https://www.kaggle.com/datasets/dgoenrique/disney-movies-and-tv-shows)  
- [HBO Max](https://www.kaggle.com/datasets/dgoenrique/hbo-max-movies-and-tv-shows)  
- [Netflix](https://www.kaggle.com/datasets/dgoenrique/netflix-movies-and-tv-shows)  
- [Paramount+](https://www.kaggle.com/datasets/dgoenrique/paramount-movies-and-tv-shows)  
- [Hulu](https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows)  

---

## 🎯 **Key Objectives**  
✔ **Analyze streaming platforms to understand content availability & trends**  
✔ **Compare genre popularity, ratings, and exclusive content across platforms**  
✔ **Develop a machine learning-based movie recommendation system**  
✔ **Visualize content distribution across different OTT services**  

---

## 📊 **Data Collection & Preprocessing**  
Each streaming platform dataset contains information on:  
- 🎞 **Title** (Movie/TV Show name)  
- 🎭 **Genre** (Action, Comedy, Drama, etc.)  
- 🎬 **Director & Cast**  
- 📅 **Release Year**  
- ⭐ **Ratings & Reviews**  
- 📺 **Platform Availability**  

✅ **Example: Loading Data from Multiple Platforms**  
```python
import pandas as pd

amazon = pd.read_csv("amazon_titles.csv")
netflix = pd.read_csv("netflix_titles.csv")
disney = pd.read_csv("disney_titles.csv")

# Combine datasets into one master dataframe
ott_content = pd.concat([amazon, netflix, disney], axis=0)
ott_content.head()
```
💡 **Why?** – This allows us to **compare platforms side by side**.  

---

## 📈 **Exploratory Data Analysis (EDA)**  
We explore trends in the streaming industry, including **genre distribution, ratings, and content exclusivity**.  

✅ **Example: Genre Distribution Across Platforms**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(y="genre", data=ott_content, order=ott_content["genre"].value_counts().index)
plt.title("Most Popular Movie Genres Across Streaming Platforms")
plt.show()
```
💡 **Insight:**  
- **Drama & Comedy dominate across platforms.**  
- **Sci-Fi & Horror are more common on Netflix & HBO Max.**  

✅ **Example: Average IMDb Ratings per Platform**  
```python
sns.boxplot(x="platform", y="imdb_rating", data=ott_content)
plt.title("IMDb Ratings Distribution by Streaming Platform")
plt.show()
```
💡 **Finding:**  
- **Netflix has the highest-rated content on average.**  
- **Amazon & Hulu have more mixed reviews.**  

✅ **Example: Content Exclusivity Analysis**  
```python
exclusive_content = ott_content.groupby("platform")["title"].nunique()
exclusive_content.plot(kind="bar", title="Exclusive Content per Platform")
```
💡 **Observation:**  
- **Disney+ and HBO Max have the highest proportion of exclusive titles.**  

---

## 🎬 **Building a Movie Recommendation System**  
We implement a **content-based recommendation system** that suggests movies based on **genre similarity**.  

✅ **Step 1: Convert Genres into Numerical Features**  
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
genre_matrix = vectorizer.fit_transform(ott_content["genre"])
```

✅ **Step 2: Compute Similarity Scores**  
```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(genre_matrix)
```

✅ **Step 3: Recommend Similar Movies**  
```python
def recommend_movie(movie_title, num_recommendations=5):
    idx = ott_content[ott_content["title"] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    recommendations = [ott_content.iloc[i[0]]["title"] for i in sorted_scores]
    return recommendations

recommend_movie("Inception", 5)
```
💡 **Example Output for "Inception":**  
1️⃣ **Interstellar**  
2️⃣ **The Matrix**  
3️⃣ **Tenet**  
4️⃣ **Blade Runner 2049**  
5️⃣ **The Prestige**  

---

## 📊 **Model Evaluation & Performance Metrics**  
To assess the **recommendation system's quality**, we use:  
✔ **Precision@K** – Measures how many recommended movies are relevant  
✔ **Diversity Score** – Ensures recommendations aren't too similar  
✔ **User Feedback Simulation** – Testing recommendations against user preferences  

✅ **Example: Evaluating Precision@K**  
```python
def precision_at_k(recommended_movies, relevant_movies, k=5):
    hits = sum(1 for movie in recommended_movies[:k] if movie in relevant_movies)
    return hits / k

# Example usage
precision_at_k(["Interstellar", "Tenet", "The Matrix"], ["Inception", "Interstellar", "The Prestige"])
```
💡 **Why?** – Higher precision means **better recommendations**.  

---

## 🔮 **Future Enhancements**  
🔹 **Hybrid Recommendation System** – Combine content-based + collaborative filtering  
🔹 **Sentiment Analysis on Reviews** – Understand audience preferences using NLP  
🔹 **Time-Series Analysis** – Predict which genres will trend in the future  
🔹 **Deploy as a Web App** – Using Flask or Streamlit for interactive recommendations  

---

## 🎯 **Why This Project Stands Out for Data Science & AI Roles**  
✔ **Combines EDA, Data Visualization & Recommendation Systems**  
✔ **Uses Real-World OTT Data for Business Insights**  
✔ **Applies Scikit-Learn & NLP for Movie Recommendations**  
✔ **Explores Cross-Platform Content Availability & Ratings**  

---

## 🛠 **How to Run This Project**  
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/streaming-wars.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook "Streaming Wars.ipynb"
   ```

---

## 📌 **Connect with Me**  
- **LinkedIn:** [Shrunali Salian](https://www.linkedin.com/in/shrunali-salian/)  
- **Portfolio:** [https://portfolio-shrunali-suresh-salians-projects.vercel.app/](#)  
- **Email:** [Your Email](#)  

---
Reference: https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system

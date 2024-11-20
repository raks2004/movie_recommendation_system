
#**MOVIE RECOMMENDATION SYSTEM**
##**INTRODUCTION**
   A movie recommendation system provides personalized movie suggestions by analyzing user data such as viewing history, ratings, preferences, and movie attributes like genre, actors, and directors. These systems are essential for platforms like Netflix, Amazon Prime, and Hulu, where users are presented with vast content libraries. By offering tailored recommendations, these systems help users discover new movies, saving them time and increasing engagement. Personalized recommendations are not only crucial for improving user satisfaction but also for enhancing user retention, as they encourage exploration of diverse genres, directors, and actors. This project aims to build a movie recommendation system that generates relevant movie suggestions with minimal user input, such as a single movie title. The system will utilize content-based filtering, which suggests movies similar to the input, and collaborative filtering, which recommends movies based on the preferences of similar users. By combining these techniques, the system will ensure diverse and accurate recommendations. Additionally, the system will address the cold-start problem, where new users or movies with limited data can still receive appropriate suggestions. The primary goal is to provide an efficient, scalable, and user-friendly recommendation system that enhances the user experience by delivering accurate and engaging movie suggestions that cater to individual tastes and preferences.

##**PROBLEM STATEMENT**
   Providing personalized movie recommendations is crucial for streaming platforms like Netflix, Amazon Prime, and Hulu because these platforms offer large libraries of content. Without a recommendation system, users would struggle to discover relevant movies among the vast number of available titles. Personalized recommendations help users find movies that match their tastes, interests, and viewing habits, enhancing user satisfaction and engagement. However, building an accurate recommendation system that tailors suggestions to individual users is challenging due to several inherent difficulties.

   ###**CHALLENGES IN RECOMMENDATION SYSTEMS**
   **Sparsity**: In recommendation systems, sparsity occurs when users don’t interact with enough items (e.g., movies) to fill the user-item matrix. This lack of data makes it harder to generate accurate recommendations, especially for new or unpopular movies.

   **Scalability**: As platforms grow, scalability becomes a challenge. The system must efficiently handle large datasets with many users and movies. Without scalability, the recommendation process can become slow and inefficient, affecting user experience.

   **Diversity**: Diversity ensures users receive varied recommendations, preventing the system from suggesting only similar content. A lack of diversity can create a “filter bubble,” limiting users’ exploration of new genres or directors.

   **Cold-start Problem**: The cold-start problem occurs when there is insufficient data for new users or movies. For new users, the system has no history to base recommendations on, and for new movies, there may be too few interactions to make accurate suggestions.

   **Bias and Popularity**: Many systems show bias toward popular movies, recommending mainstream content over niche options. This can limit the system’s ability to offer personalized suggestions that truly match users' unique tastes.
##**OBJECTIVE**
   The objective of this project is to develop a movie recommendation system that provides personalized suggestions based on minimal input, such as a single movie title or genre. The system will use a hybrid approach, combining content-based filtering, which analyzes movie attributes like genre, director, and cast, and collaborative filtering, which leverages user preferences to recommend movies liked by similar users. To address the cold-start problem, genre-based recommendations will be offered for new users or movies. The system will be built using the TMDB 5000 movie dataset and will focus on algorithm development with machine learning techniques to ensure accuracy and diversity in recommendations. Performance will be evaluated through metrics like accuracy and user satisfaction, with user feedback incorporated to improve recommendations over time. This project aims to enhance learning in machine learning and recommendation algorithms while delivering an efficient and scalable recommendation system.
##**DATASET**
   The dataset used for this project is the TMDB 5000 Movie Dataset, sourced from Kaggle. It consists of two CSV files that provide detailed metadata about movies and their associated credits. The dataset offers a comprehensive foundation for building a recommendation system due to its rich set of attributes.

   **tmdb\_5000\_credits.csv**:
   This file contains information about the cast and crew involved in each movie, along with the movie ID and title.

Columns:

- **movie\_id**: Unique identifier for each movie.
- **title**: The title of the movie.
- **cast**: A list of cast members in JSON format.
- **crew**: A list of crew members (e.g., director, producer) in JSON format.

**tmdb\_5000\_movies.csv**:
This file provides additional details about movies, including their attributes, popularity, and genres.

Columns:

- **budget**: The production budget of the movie.
- **genres**: A list of genres in JSON format.
- **homepage**: The official homepage of the movie (if available).
- **id**: A unique identifier matching the movie\_id in the credits file.
- **keywords**: Keywords describing the movie in JSON format.
- **original\_language**: The primary language of the movie.
- **original\_title**: The original title of the movie.
- **overview**: A short summary or description of the movie.
- **popularity**: A numeric score representing the movie's popularity.
- **production\_companies**: A list of production companies involved in the movie in JSON format.
- **production\_countries**: A list of countries where the movie was produced, in JSON format.
- **release\_date**: The release date of the movie.
- **revenue**: Total revenue generated by the movie.
- **runtime**: Duration of the movie in minutes.
- **spoken\_languages**: A list of languages spoken in the movie, in JSON format.
- **status**: The release status of the movie (e.g., Released, Post-Production).
- **tagline**: A promotional tagline for the movie.
- **title**: The title of the movie (same as original title).
- **vote\_average**: The average user rating.
- **vote\_count**: The total number of user votes.

The dataset is particularly suitable for building a movie recommendation system due to the diversity of its features. The genres, cast, and keywords enable content-based filtering, while user ratings, popularity, and vote counts can be used for collaborative filtering. Columns like production\_countries and spoken\_languages provide additional metadata that could enhance recommendations' diversity and accuracy.
##**METHODOLOGY**
   ###**DATA PREPROCESSING**
   For the data preprocessing phase, the dataset was first imported using pandas and ast to handle the CSV files and parse any JSON-formatted columns. The two CSV files, tmdb\_5000\_credits.csv and tmdb\_5000\_movies.csv, were merged based on the movie\_id column, ensuring a unified dataset for analysis. Irrelevant columns such as budget, homepage, original\_title, production\_companies, production\_countries, revenue, status, tagline, spoken\_languages, and runtime were dropped to focus on the most pertinent data for the recommendation system. Missing values were addressed by filling the overview column with the placeholder text 'No description available' and the release\_date column with 'Unknown'. JSON-formatted columns like genres, keywords, and cast were parsed into lists to simplify data processing. To further streamline the dataset and focus on relevant attributes for building the recommendation system, unnecessary columns such as original\_language, popularity, release\_date, vote\_count, and crew were removed, retaining only essential columns like genres, movie\_id, keywords, overview, vote\_average, title, and cast. Finally, the cleaned dataset was saved as preprocessed\_movies.csv, ready for further feature engineering and model development.



###**FEATURE ENGINEERING**
In the feature engineering phase, the preprocessed dataset was further refined to extract meaningful numerical features from the available data. The genres column was transformed using multi-hot encoding, representing each genre as a binary feature. For the keywords column, TF-IDF vectorization was applied to capture the importance of keywords across movies. The overview column containing textual data was converted into numerical features using pre-trained word embeddings, generating dense vectors that encapsulated semantic meaning. The vote\_average column, a numerical feature, was normalized using Min-Max Scaling to ensure uniformity with other features. To address the high dimensionality introduced by multi-hot encoding and TF-IDF, Principal Component Analysis (PCA) was employed to reduce the dimensional space while retaining maximum variance. The final dataset consisted of numerical features like overview\_embedding, PCA components (PC1, PC2, etc.), and reference columns such as movie\_id, title, and overview, making it ready for the modeling phase.
##
##**MODELING**
   The recommendation system employed Manhattan distance as the similarity metric. This metric calculates the "absolute distance" between two vectors, making it suitable for comparing high-dimensional data. Each movie's composite vector was constructed by concatenating its overview\_embedding and PCA components, ensuring that semantic and metadata information contributed to the similarity calculations.

   When a user inputs a movie title, the system retrieves the corresponding composite vector and computes Manhattan distances between this vector and all other movie vectors in the dataset. The movies are then ranked by distance, with smaller distances indicating higher similarity. To avoid redundancy, the input movie itself is excluded from the results. Finally, the system outputs the top-N recommendations (default N=10), providing relevant movie suggestions.
   ##
##**FUTURE ENHANCEMENTS**
   While the current system is robust, its functionality can be enhanced through user feedback mechanisms. For example, incorporating "like" or "dislike" feedback for recommended movies can allow the system to adapt and improve over time. This feedback could be stored and used to refine the model periodically, ensuring the system avoids recommending disliked movies in the future. Additionally, integrating collaborative filtering could complement the content-based approach. Combining user interaction data with existing features would enhance the diversity and personalization of recommendations, making the system more dynamic and user-centric.


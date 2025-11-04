# **Steam Market Insights & Data Advisor**

An app to predict Steam game success, providing market insights and advising developers using machine learning.

This project showcases a complete data science pipeline: data collection (scraping), data cleaning, feature engineering, exploratory data analysis (EDA), model training, and deployment as an interactive Streamlit web application.

## **Live Application**

[Streamlit App](https://steam-market-insights-data-advisor.streamlit.app/)[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steam-market-insights-data-advisor.streamlit.app/)

## **Project Workflow**

1. **Data Collection:** Scraped data for \~5,000 top games from the Steam Store, Steam Review API, and SteamCharts using requests and BeautifulSoup.  
2. **Feature Engineering:** Cleaned the raw data and engineered custom metrics to define "Success" and "Longevity" in a quantifiable way.  
3. **Predictive Modeling:** Trained two XGBoost classification models to predict the engineered SuccessTier and LongevityTier based on pre-launch features.  
4. **Web Application:** Built an interactive Streamlit app (app.py) for developers to get predictions, actionable insights (using SHAP), and market analysis for their game concepts.

## **1\. Data Collection**

Data was scraped from multiple sources to create a comprehensive dataset. The following features were collected for each game:

* **From Steam Store:**  
  * Name: Game Title  
  * Price (USD): Current price in USD  
  * Is Free: Boolean (True/False)  
  * Release Date: Original release date  
  * Tags: All user-defined tags (e.g., "Indie", "Action", "RPG")  
  * Controller Support: Boolean (True/False)  
  * Steam Deck Support: Boolean (True/False)  
  * Languages: List of supported languages  
* **From Steam Review API:**  
  * Review Summary: Text descriptor (e.g., "Very Positive")  
  * Total Reviews: Total number of user reviews  
  * Positive Reviews: Number of positive reviews  
* **From SteamCharts:**  
  * All-Time Peak Players: Peak concurrent players  
  * Avg Players (30 Days): Average concurrent players in the last 30 days

## **2\. Data Preprocessing & Feature Engineering**

This is the core of the project. Raw data was cleaned (handling missing prices, parsing dates), and several key features were engineered to be used as modeling targets and inputs.

### **Engineered Target Variables**

Simple sales or review scores are not enough. "Success" is a combination of quality (rating) and quantity (sales volume), while "Longevity" is a measure of sustained engagement over time.

#### **1\. Review Success Score (Wilson Score)**

Instead of a simple ratio (Positive / Total), which is unreliable for games with few reviews, I used the **Wilson Score Confidence Interval**. This provides a statistically robust score of quality, balancing the positive review ratio with the total number of reviews.

ReviewSuccessScore \= calculate\_wilson\_score(positive\_reviews, total\_reviews)

#### **2\. Commercial Success Score (Target)**

This is the primary metric for success. It combines the *quality* of the game (Wilson Score) with its *popularity/volume* (log of total reviews). This prevents games with millions of "Mixed" reviews from being scored as highly as a game with thousands of "Overwhelmingly Positive" reviews.

CommercialSuccessScore \= ReviewSuccessScore \* $\log_{10}$(TotalReviews \+ 1\)

#### **3\. Longevity Score (Target)**

This metric measures a game's ability to retain players long after release. It's a product of the log of the average player base and the log of the game's age. This rewards games that maintain a stable player count over many years.

LongevityScore \= $\log_{10}$(AvgPlayers30Days \+ 1\) \* $\log_{10}$(DaysSinceRelease \+ 1\)

### **Engineered Input Features**

These features were created from the raw data to be used as inputs for the model.

* **LanguageMarketScore**: Calculated the potential market reach by summing the global percentage of Steam users for each language the game supports (e.g., English: 35.62%, Simplified Chinese: 26.03%).  
* **IsMultiplayer**: A boolean feature (1 or 0\) derived by checking if a game's tags contained any multiplayer-related keywords (e.g., 'Multiplayer', 'MMO', 'Co-op').

## **3\. Exploratory Data Analysis (EDA)**

Analysis (exploratory\_data\_analysis.ipynb) was performed to understand the relationships between features.

* An estimated sales figure was created using the "Boxleiter number" (EstimatedSales \= TotalReviews \* 40).  
* A correlation heatmap showed a strong positive correlation between our engineered CommercialSuccessScore and All-Time Peak Players, validating it as a strong proxy for success.  
* Price showed a complex, non-linear relationship with success metrics, with many high-performing games existing in the 10$-30$ range, but also a large cluster of "Blockbuster" F2P titles.

<img width="1329" height="1144" alt="image" src="https://github.com/user-attachments/assets/09c0fef9-da16-4fb5-bc0c-d9eb01a9e511" />


## **4\. Predictive Modeling**

Two classification models were trained to predict the *tier* of success a game would achieve, based only on features known *before* launch.

* **Model**: XGBoost Classifier  
* **Targets (Y)**:  
  1. SuccessTier: The CommercialSuccessScore binned into 6 categories ('Niche or Flop', 'Modest Success', 'Solid Performer', 'Notable Hit', 'Major Hit', 'Blockbuster').  
  2. LongevityTier: The LongevityScore binned into 4 categories ('Fades Quickly', 'Average Lifespan', 'High Retention', 'Evergreen').  
* **Features (X)**:  
  * Price (USD)  
  * Controller Support (bool)  
  * Steam Deck Support (bool)  
  * IsMultiplayer (bool)  
  * LanguageMarketScore (float)  
  * Tags (One-Hot Encoded from the Top 50 most common tags)

### **Model Explainability (SHAP)**

To make the models' predictions useful, **SHAP (SHapley Additive exPlanations)** was implemented. This allows the app to not only give a prediction but *explain why* it made that prediction. It identifies which features (like specific tags or price) had the biggest positive or negative impact on the success score.

<img width="1907" height="848" alt="image" src="https://github.com/user-attachments/assets/1df5f3d4-96fe-4736-ad0c-1dbdc2651f30" />

## **5\. The Streamlit Application**

The final models and data pipeline are served in an interactive Streamlit app (app.py).

### **Key App Features:**

* **Success Prediction**: Users input their game's pre-launch features, and the app provides probability charts for both SuccessTier and LongevityTier.  
* **Actionable Tag Recommendations**: Using SHAP values, the app analyzes which *unselected* tags would provide the biggest potential boost to the game's predicted success score.  
* **Competitive Benchmark**: The app finds games in the database with similar tags (using TF-IDF for tag importance) and shows where the user's "Your Game" would rank among them based on its predicted score.  
* **Pricing Insights**: Provides a bar chart of competitor pricing and shows the breakdown of similar games that are Free to Play vs. Paid.

## **How to Run Locally**

1. Clone the repository:  
   git clone https://github.com/FilipDopierala/Steam-Market-Insights-Data-Advisor  
   cd Steam-Market-Insights-Data-Advisor

2. Install the required dependencies:  
   pip install \-r requirements.txt
 
3. Run the Streamlit app:  
   streamlit run app.py  

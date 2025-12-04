Predicting User Engagement Ratio on YouTube: MrBeast Case Study

📌 Project Overview

This project presents a comprehensive machine learning analysis aimed at predicting the User Engagement Ratio on YouTube, specifically defining engagement as the Like-to-View ratio rather than traditional view counts. Using the MrBeast channel as a primary case study, this research seeks to isolate genuine audience interaction from algorithmic virality. The MrBeast channel, known for its extreme outliers and high-budget content, provides a unique, high-volume dataset ideal for testing predictive models on "viral" behavior.

The core problem addressed is that raw view counts can often be misleading metrics of quality, frequently driven by clickbait or "algorithmic luck." In contrast, the Like/View ratio offers a purer signal of content quality and audience appreciation. To explore this, the project adopts a dual methodological approach:

Classification: A framework that categorizes videos into three distinct levels of success—Minimum, Medium, and Viral—based on statistical percentiles to understand engagement thresholds.

Regression: A predictive model designed to forecast the precise numerical value of the engagement ratio, allowing for granular performance analysis.

Data was harvested using the Google YouTube Data API v3, extracting granular details from 400 videos. The preprocessing pipeline involved rigorous cleaning and Feature Engineering, including the creation of temporal variables like VideoAge and interaction metrics such as the Comments/ViewsRatio. A RobustScaler was applied to normalize the data, a critical step given the significant outliers inherent in social media data.

The analysis utilized a variety of algorithms, ranging from Linear Regression and KNN to ensemble methods like Random Forest and XGBoost. The results conclusively demonstrated the superiority of tree-based models, with Random Forest and XGBoost achieving classification accuracies between 65% and 75% and identifying low-engagement videos with an AUC of 0.92. A key insight from the Exploratory Data Analysis was the strong positive correlation between comment density and like ratios, suggesting that interaction is cumulative. Ultimately, this project highlights the predictability of engagement metrics while acknowledging the inherent challenges in distinguishing mid-tier performance in highly viral environments.

📂 Data Gathering & Engineering

Data Source: Extracted data from 400 videos using the Google YouTube Data API v3.

Preprocessing: Handled date formats, removed unnecessary columns, and standardized features using RobustScaler to handle outliers.

Feature Engineering:

VideoAge: Days since publication.

Like/ViewsRatio: Target variable (nLikes / nViews).

Comments/ViewsRatio: Interaction index.

🚀 Methodology

1. Classification

The dataset was split into training (75%) and testing (25%) sets. Videos were labeled into three engagement classes based on percentiles.

Models Used: Random Forest, KNN, Naive Bayes, XGBoost, SVM.

Metrics: Accuracy, Precision, Recall, F1-Score, ROC/AUC.

2. Regression

Predicted the exact engagement score using various regression techniques.

Models Used: Linear Regression, Lasso, Random Forest, Decision Tree (CART), XGBoost, SVR, KNN.

Metrics: MSE (Mean Squared Error), MAE (Mean Absolute Error), R2 Score.

📊 Key Results

Dominant Models: Random Forest and XGBoost outperformed all other models in both classification and regression tasks.

Performance: Achieved classification accuracy between 65% - 75%, with high AUC scores for identifying "Minimum" and "Viral" videos.

Insights: A strong correlation was found between the Comment/View ratio and the Like/View ratio, suggesting that interaction breeds further engagement.

🛠️ Technologies Used

Language: Python

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly, WordCloud

Machine Learning: Scikit-Learn, XGBoost

API: Google API Client (YouTube Data API v3)

NLP: NLTK (for title analysis)

🔧 Installation & Usage

Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)



Install dependencies:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn google-api-python-client plotly nltk wordcloud



Set up your YouTube Data API key in the script variables.

Run the Jupyter Notebook or Python script to perform the analysis.

👤 Author

Daniele Mariani

Email: danyelemariani@gmail.com

University: University of Milano-Bicocca

This project was developed for the Workshop Programming course.

#  Customer Journey Analysis Using Clustering and Dimensionality Reduction
This project aims to analyze customer behavior in the tourism industry by using machine learning techniques like clustering and dimensionality reduction. The analysis helps segment customers based on their interactions, preferences, and demographics, which can assist in targeted marketing and personalized recommendations.

📌 Overview
This project aims to analyze customer behavior in the tourism sector using machine learning techniques to derive actionable insights.
The dataset consists of 11,760 records and 17 columns, combining 10 numerical and 7 categorical variables.

The target variable is Taken_product, indicating whether a customer opted for a tourism-related product or service.

🎯 Key Objectives
Perform comprehensive Data Cleaning and Exploratory Data Analysis (EDA)

Handle missing and duplicate values appropriately

Apply Principal Component Analysis (PCA) for dimensionality reduction

Segment customers using KMeans Clustering

Visualize and profile clusters to derive business insights

🛠️ Technologies & Libraries Used
Python – Core programming language

Pandas, NumPy – Data loading and preprocessing

Matplotlib, Seaborn, Plotly – Visualization

Scikit-learn – Machine learning: PCA, KMeans, and evaluation metrics

Streamlit – Interactive dashboard (for cluster analysis demo)

Yellowbrick (optional) – Visual analysis of clusters

📊 Key Features
🧹 Data Cleaning
Categorical missing values imputed using mode

Numeric missing values handled using median

Device categories were standardized

Outliers were capped using the 5th and 95th percentiles

📈 Exploratory Data Analysis
Univariate and bivariate analysis of numeric and categorical features

Visualizations of distributions, trends, and correlations

Custom donut and bar charts for categorical feature exploration

🔽 Dimensionality Reduction
PCA reduced the complexity of feature space while preserving maximum variance

Top 2 components used for cluster visualization

📊 Clustering
KMeans Clustering applied on standardized and reduced data

Elbow Method and Silhouette Score used to choose optimal k

Customers segmented into groups like:

“Highly Engaged”

“Moderately Engaged”

“Low Engagement”

📌 Visualization
PCA scatter plots to show separation between clusters

Cluster summary with mean feature values per group

Streamlit app for live interactive segmentation

📁 Dataset Highlights
Behavioral features:
Yearly_avg_view_on_travel_page, total_likes_on_outstation_checkin_given, Daily_Avg_mins_spend_on_traveling_page, etc.

Demographics & user traits:
preferred_device, working_flag, member_in_family, Adult_flag, etc.

🚀 Results
Identified 3–5 customer segments with distinct behavioral patterns.

Segments helped categorize customers into profiles like:

Active travelers

Casual browsers

Potential leads

These insights can guide:

Targeted marketing

Customized travel packages

Retention campaigns

UX personalization

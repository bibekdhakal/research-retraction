# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.metrics import r2_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from mpl_toolkits.mplot3d import Axes3D
# import datetime

# # Load the dataset
# data = pd.read_csv("../new_retractions.csv")

# # Convert date columns to datetime objects
# data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], format='%d/%m/%Y', errors='coerce')
# data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], format='%d/%m/%Y', errors='coerce')

# # Drop rows with missing or incorrect dates
# data = data.dropna(subset=['OriginalPaperDate', 'RetractionDate'])

# # Data Exploration and Pre-processing
# # Summary Statistics
# summary_stats = data.describe()

# # Variable Distributions
# data.hist(figsize=(10,10))
# plt.show()

# # Bar Charts for Categorical Variables
# categorical_vars = ['Subject_category', 'ArticleType_category', 'Publisher_category', 'Reason_category', 'Paywalled_category']
# for col in categorical_vars:
#     sns.countplot(x=col, data=data)
#     plt.show()

# # Identify and handle missing data
# missing_data = data.isnull().sum()

# # Analyze temporal trends
# data['Year'] = data['OriginalPaperDate'].dt.year
# temporal_trends = data.groupby('Year').size()

# # Explore subject-wise distribution of retractions
# subject_distribution = data['Subject_category'].value_counts()

# # Assess correlations between variables
# correlation_matrix = data.corr()

# # Regression
# # Forward selection and backward elimination for feature selection

# # Split data into features and target
# X = data.drop(columns=['RetractionDate', 'Record ID']) # Features
# y = data['RetractionDate'] # Target

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Evaluate model performance
# y_pred = model.predict(X_test)
# r_squared = r2_score(y_test, y_pred)

# # Plot R-squared versus number of features
# num_features = range(1, len(X.columns)+1)
# r_squared_values = []
# for i in num_features:
#     model.fit(X_train.iloc[:, :i], y_train)
#     y_pred = model.predict(X_test.iloc[:, :i])
#     r_squared_values.append(r2_score(y_test, y_pred))

# plt.plot(num_features, r_squared_values)
# plt.xlabel('Number of Features')
# plt.ylabel('R-squared')
# plt.title('R-squared vs. Number of Features')
# plt.show()

# # Dimensionality Reduction with PCA
# # Principal Component Analysis (PCA)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA()
# X_pca = pca.fit_transform(X_scaled)

# # Explained Variance Ratio
# explained_variance_ratio = pca.explained_variance_ratio_

# # Regression with PCA
# X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# model_pca = LinearRegression()
# model_pca.fit(X_train_pca, y_train_pca)

# y_pred_pca = model_pca.predict(X_test_pca)
# r_squared_pca = r2_score(y_test_pca, y_pred_pca)

# # Improving PCA
# # Standardization and Min-Max Scaling
# scaler_minmax = MinMaxScaler()
# X_scaled_minmax = scaler_minmax.fit_transform(X)

# # Data Distribution Comparison
# # Visualizing Data Distributions
# plt.figure(figsize=(10, 6))
# sns.histplot(data['CitationCount'], kde=True, color='blue', label='Original')
# sns.histplot(data['CitationCount'].apply(np.log1p), kde=True, color='red', label='Log-transformed')
# plt.title('Data Distribution Comparison')
# plt.xlabel('Citation Count')
# plt.legend()
# plt.show()

# # PCA Visualization
# # Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(data.drop(columns=['RetractionDate', 'Record ID']))

# # Apply PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Plot PCA visualization
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
# plt.title('PCA Visualization')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

# # K-Means Clustering
# # K-means clustering without standardization
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(data.drop(columns=['RetractionDate', 'Record ID']))

# # K-means clustering with standardization
# kmeans_scaled = KMeans(n_clusters=3)
# kmeans_scaled.fit(X_scaled)

# # Find the optimal k parameter using the Elbow Method
# inertia = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X_scaled)
#     inertia.append(kmeans.inertia_)

# plt.figure(figsize=(8, 6))
# plt.plot(range(1, 11), inertia, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.show()

# # Visualize the natural groupings of data points versus the clusters found by k-means
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_scaled.labels_, palette='viridis')
# plt.title('K-means Clustering with Standardization')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

# # Use k-means to classify objects automatically
# predicted_clusters = kmeans_scaled.predict(X_scaled)

# # Visualize the true class vs. class predicted by k-means
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Subject_category'], palette='viridis')
# plt.title('True Class vs. Predicted Class by K-means')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("../new_retractions.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data['Subject_category'] = label_encoder.fit_transform(data['Subject_category'])
data['Reason_category'] = label_encoder.fit_transform(data['Reason_category'])

# Prepare features and target
X = data[['Subject_category', 'Reason_category']]
y = data['Publisher_category']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# K-means clustering without standardization
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
y_pred_train = kmeans.predict(X_train)
y_pred_test = kmeans.predict(X_test)

# Evaluate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(12, 8))  # Increase figure size
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, cbar=False)  # Hide color bar
plt.xlabel('Predicted', fontsize=12)  # Increase font size
plt.ylabel('True', fontsize=12)  # Increase font size
plt.title('Confusion Matrix', fontsize=14)  # Increase title font size
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.show()

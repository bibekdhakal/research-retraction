

# If author affiliation information is available (e.g., university, institution), you can group retractions by affiliation and calculate retraction rates.
# This might reveal potential biases or areas needing improvement within specific institutions.
# Additional Analyses:

# You can explore relationships between retraction reasons and other factors like publication type, journal impact factor (if available), or the number of authors.
# Use techniques like correlation analysis or statistical tests to see if there are significant associations.

# Regression: Regression models can be used to predict numerical outcomes, such as the number of retractions or the time until the next retraction. For example, you could use linear regression to predict the number of retractions based on various predictor variables, such as publication year, journal impact factor, author reputation, etc.

# Clustering: Clustering algorithms can help identify groups or clusters of publications with similar characteristics. This can be useful for identifying patterns or trends in retraction data. For example, you could use k-means clustering to group publications based on features such as publication type, research methodology, or citation count. This could reveal common characteristics among retractions and potentially uncover underlying reasons for retractions.

# Classification: Classification algorithms can be used to predict categorical outcomes, such as whether a publication is likely to be retracted or not. For example, you could train a binary classification model (e.g., logistic regression, decision tree, random forest) using features such as author reputation, journal impact factor, citation count, etc., to classify publications as either high-risk (likely to be retracted) or low-risk (unlikely to be retracted).


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

def visualize_article_types_distribution_yearwise(data):
    # Split ArticleType column by semicolon and explode into separate rows
    data['article_types'] = data['ArticleType'].str.split(';')
    data = data.explode('article_types')

    # Parse RetractionDate column with multiple date formats
    data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

    # Extract year from RetractionDate
    data['year'] = data['RetractionDate'].dt.year

    # Drop rows with NaN in year (missing or incorrect RetractionDate)
    data = data.dropna(subset=['year'])

    # Group by year and article type, count occurrences
    article_type_counts_per_year = data.groupby(['year', 'article_types']).size().unstack(fill_value=0)

    # Plot distribution of article types over years
    article_type_counts_per_year.plot(kind='line', cmap='tab20', figsize=(10, 6))
    plt.xlabel("Retracted Year")
    plt.ylabel("Number of Articles")
    plt.title("Distribution of Article Types by Retraction Year (Lines)")
    plt.legend(title="Article Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def analyze_retractions_by_reason(data):
    # Check if there is a column describing the reason for retraction
    if 'Reason' not in data.columns:
        print("Error: Reason for retraction column not found.")
        return

    # Count the frequency of each reason for retraction
    reason_counts = data['Reason'].str.split(';').explode().str.strip().dropna().value_counts()

    # Identify the most frequent reasons 
    most_frequent_reasons = reason_counts.head(15)

    # Plot a pie chart to visualize the distribution of reasons
    plt.figure(figsize=(8, 8))
    plt.pie(most_frequent_reasons, labels=most_frequent_reasons.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Retractions by Reason')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()

    # Show legends by color
    plt.legend(loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1), title="Reasons", fontsize='small', fancybox=True, shadow=True)
    plt.show()

def analyze_retractions_by_country(data):
    try:
        # Check if there is a column describing the country for each retraction
        if 'Country' not in data.columns:
            raise KeyError("Country column not found in the DataFrame.")

        # Split entries containing multiple countries and exclude 'Unknown' countries
        data['Country'] = data['Country'].str.replace(',', ';')  # Replace commas with semicolons
        data['Country'] = data['Country'].str.split(';').apply(lambda x: [c.strip() for c in x if c.strip() != 'Unknown'])
        data = data.explode('Country')
        
        # Print the first few rows of the 'Country' column after manipulation
        print("\nAfter splitting and exploding:")
        print(data['Country'].head())

        # Count the number of retractions for each country
        country_counts = data['Country'].value_counts()
        print(country_counts)
        # Select the top 10 countries
        top_countries = country_counts.head(10)

        # Plot a pie chart to visualize the distribution of retractions by country
        plt.figure(figsize=(8, 8))
        plt.pie(top_countries, labels=top_countries.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Retractions by Country')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()

        # Load the world shapefile
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Print the data types of the columns in the world shapefile
        print("COlumns:\n",world.columns)
        print('Country\n',country_counts.index)

        # Merge country counts with world shapefile
        world = world.merge(country_counts, left_on='name', right_index=True)
        print('Merged data\n',world)
        print('Count word: ',world['count'].dtype)


        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world.plot(column='count', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
        ax.set_title('Retractions by Country')
        plt.show()

    except KeyError as e:
        print("Error:", e)

def perform_linear_regression(data):
    # Split ArticleType column by semicolon and explode into separate rows
    data['article_types'] = data['ArticleType'].str.split(';')
    data = data.explode('article_types')

    # Parse RetractionDate column with multiple date formats
    data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')

    # Extract year from RetractionDate
    data['year'] = data['RetractionDate'].dt.year

    # Drop rows with NaN in year (missing or incorrect RetractionDate)
    data = data.dropna(subset=['year'])

    # Combine reasons and article types into a single column for encoding
    data['features'] = data['Reason'] + ';' + data['article_types']

    # Initialize MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer()

    # One-hot encode the combined features
    features_encoded = pd.DataFrame(mlb.fit_transform(data['features'].str.split(';')),
                                    columns=mlb.classes_,
                                    index=data.index)

    # Concatenate encoded features with retraction year
    X = pd.concat([features_encoded, data['year']], axis=1)
    y = data['year']  # Target variable is the retraction year

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    mse = model.score(X_test, y_test)
    print('Mean Squared Error:', mse)

    # Print the coefficients of the regression model
    print('Coefficients:', model.coef_)


# Load the data from a CSV file
data = pd.read_csv("../retractions35215.csv")

# Call the function to visualize article types distribution over time
# visualize_article_types_distribution_yearwise(data)

# Analyze retractions by reason
# analyze_retractions_by_reason(data)

# Analyze retractions by country
# analyze_retractions_by_country(data)

# Call the function with your dataset
perform_linear_regression(data)

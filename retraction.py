# 1. Analyze Trends in Retractions Over Time:

# Look for a column indicating the publication or retraction date. You can use this to:
# Calculate the yearly retraction rate (number of retractions per year divided by the total number of publications in that year).
# Plot a time series graph to visualize how retraction rates have changed over time.
# 2. Analyze Retractions by Publication Type:

# Identify a column categorizing the type of publication (e.g., journal article, conference proceeding, book chapter).
# Calculate retraction rates for different publication types and compare them.
# 3. Analyze Retractions by Reason:

# Look for a column describing the reason for retraction (if available).
# Identify the most frequent reasons for retractions (e.g., misconduct, honest errors).
# You can visualize this using bar charts or pie charts.
# 4. Analyze Retractions by Country:

# If a country of origin is included for the corresponding authors, you can group retractions by country and calculate retraction rates.
# Analyze if there are any geographical trends in retraction rates.
# 5. Analyze Retractions by Author Affiliation:

# If author affiliation information is available (e.g., university, institution), you can group retractions by affiliation and calculate retraction rates.
# This might reveal potential biases or areas needing improvement within specific institutions.
# Additional Analyses:

# You can explore relationships between retraction reasons and other factors like publication type, journal impact factor (if available), or the number of authors.
# Use techniques like correlation analysis or statistical tests to see if there are significant associations.
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

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


# Load the data from a CSV file
data = pd.read_csv("../retractions35215.csv")

# Call the function to visualize article types distribution over time
visualize_article_types_distribution_yearwise(data)

# Analyze retractions by reason
analyze_retractions_by_reason(data)

# Analyze retractions by country
analyze_retractions_by_country(data)

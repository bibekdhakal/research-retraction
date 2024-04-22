import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\pavan\Documents\Data Visualization aiisgnment 1\retractions35215.csv")
# print(df.head())
df.info()
df.describe()
print(df['Subject'].isnull().sum())
subject_counts = df.groupby('Subject').size()
print(subject_counts)
# number of occurences unique to subject values
 
subject_counts = df['Subject'].value_counts()
num_unique_subjects = len(subject_counts)
print("Number of unique subjects:", num_unique_subjects)
print("Counts of each unique subject:")
print(subject_counts)

# Count the occurrences of each unique value in the 'Country' column and also count the number of unique values
country_counts = df['Country'].value_counts()
num_unique_countries = len(country_counts)
print("Number of unique countries:", num_unique_countries)
print("Counts of each unique country:")
print(country_counts)

# Check for null values in the 'Publisher' column
print("Number of null values in 'Publisher' column:", df['Publisher'].isnull().sum())

# Count the occurrences of each unique value in the 'Publisher' column and also count the number of unique values
publisher_counts = df['Publisher'].value_counts()
num_unique_publishers = len(publisher_counts)
print("Number of unique publishers:", num_unique_publishers)
print("Counts of each unique publisher:")
print(publisher_counts)

# Check for null values
print("Number of null values in 'ArticleType' column:", df['ArticleType'].isnull().sum())
print("Number of null values in 'RetractionNature' column:", df['RetractionNature'].isnull().sum())

# Count the occurrences of each unique value
article_type_counts = df['ArticleType'].value_counts()
retraction_nature_counts = df['RetractionNature'].value_counts()

# Count the number of unique values
num_unique_article_types = len(article_type_counts)
num_unique_retraction_nature = len(retraction_nature_counts)

# Print the counts
print("Number of unique article types:", num_unique_article_types)
print("Counts of each unique article type:")
print(article_type_counts)

# Subject vs. Country
subject_country_table = pd.crosstab(df['Subject'], df['Country'])

# Subject vs. Publisher
subject_publisher_table = pd.crosstab(df['Subject'], df['Publisher'])

# Subject vs. ArticleType
subject_article_type_table = pd.crosstab(df['Subject'], df['ArticleType'])

print("Subject vs. Country:")
print(subject_country_table)
print("\nSubject vs. Publisher:")
print(subject_publisher_table)
print("\nSubject vs. ArticleType:")
print(subject_article_type_table)














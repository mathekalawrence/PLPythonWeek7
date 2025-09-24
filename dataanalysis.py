# Data Analysis and Visualization
# Preferred Dataset: Iris Flower Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Setting styles for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_dataset():
    """
    Task 1: Loading and Exploring the Dataset
    """
    print("=" * 60)
    print("TASK 1: LOADING AND EXPLORING THE DATASET")
    print("=" * 60)
    
    try:
        # Loading the Iris dataset from sklearn
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        
        # Displaying the first few rows
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        # Exploring the dataset structure
        print("\nDataset information:")
        print(df.info())
        
        # Checking for the missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        # Since Iris dataset is clean, here is the cleaning process demonstration
        print("\nNo missing values found. Dataset is clean!")
        
        return df
        
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return None

def basic_data_analysis(df):
    """
    Task 2: Basic Data Analysis
    """
    print("\n" + "=" * 60)
    print("TASK 2: BASIC DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics for numerical columns
    print("Basic statistics for numerical columns:")
    print(df.describe())
    
    # Grouping by species and compute mean for numerical columns
    print("\nMean values grouped by species:")
    species_group = df.groupby('species').mean()
    print(species_group)
    
    # Additional analysis: Finding patterns
    print("\n" + "-" * 40)
    print("MORE INTERESTING FINDINGS:")
    print("-" * 40)
    
    # FindING which species has the largest petals
    max_petal_length = df.loc[df['petal length (cm)'].idxmax()]
    print(f"Largest petal length: {max_petal_length['petal length (cm)']} cm ({max_petal_length['species']})")
    
    # Finding which species has the smallest sepals
    min_sepal_width = df.loc[df['sepal width (cm)'].idxmin()]
    print(f"Smallest sepal width: {min_sepal_width['sepal width (cm)']} cm ({min_sepal_width['species']})")
    
    # Correlation analysis
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    correlation_matrix = df[numerical_cols].corr()
    print("\nCorrelation matrix:")
    print(correlation_matrix)
    
    return species_group

def create_visualizations(df, species_group):
    """
    Task 3: Data Visualization
    """
    print("\n" + "=" * 60)
    print("TASK 3: DATA VISUALIZATION")
    print("=" * 60)
    
    # Creating a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Flower Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Visualization 1: Line chart (simulated time series)
    # Since Iris dataset doesn't have time, i'll use index as pseudo-time; challenging though!
    print("Creating Visualization 1: Line Chart (Pseudo-time series)...")
    axes[0, 0].plot(df.index[:50], df['sepal length (cm)'][:50], marker='o', label='Sepal Length', linewidth=2)
    axes[0, 0].plot(df.index[:50], df['petal length (cm)'][:50], marker='s', label='Petal Length', linewidth=2)
    axes[0, 0].set_title('Trend of Sepal and Petal Length (First 50 Samples)')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Length (cm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Visualization 2: Bar chart - Average measurements by species
    print("Creating Visualization 2: Bar Chart...")
    species_means = species_group.T  # Transpose for better visualization
    x = np.arange(len(species_means.columns))
    width = 0.2
    
    for i, measurement in enumerate(species_means.index):
        axes[0, 1].bar(x + i*width, species_means.iloc[i], width, label=measurement)
    
    axes[0, 1].set_title('Average Measurements by Iris Species')
    axes[0, 1].set_xlabel('Species')
    axes[0, 1].set_ylabel('Measurement (cm)')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(species_means.columns)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Visualization 3: Histogram - Distribution of sepal length
    print("Creating Visualization 3: Histogram...")
    species_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species]['sepal length (cm)']
        axes[1, 0].hist(species_data, alpha=0.7, label=species, color=species_colors[species], bins=15)
    
    axes[1, 0].set_title('Distribution of Sepal Length by Species')
    axes[1, 0].set_xlabel('Sepal Length (cm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Visualization 4: Scatter plot - Sepal length vs Petal length
    print("Creating Visualization 4: Scatter Plot...")
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        axes[1, 1].scatter(species_data['sepal length (cm)'], 
                          species_data['petal length (cm)'], 
                          label=species, 
                          alpha=0.7,
                          s=60)
    
    axes[1, 1].set_title('Sepal Length vs Petal Length')
    axes[1, 1].set_xlabel('Sepal Length (cm)')
    axes[1, 1].set_ylabel('Petal Length (cm)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjusting layout and saving the figure
    plt.tight_layout()
    plt.savefig('iris_data_analysis.png', dpi=300, bbox_inches='tight')
    print(" All visualizations created successfully!")
    print(" Plot saved as 'iris_data_analysis.png'")
    
    # Showing the plots
    plt.show()
    
    # Additional advanced visualization using seaborn
    print("\nCreating additional visualization: Pair Plot...")
    sns.pairplot(df, hue='species', diag_kind='hist', palette='husl')
    plt.suptitle('Pair Plot of Iris Dataset Features', y=1.02)
    plt.savefig('iris_pair_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to execute all tasks
    """
    print("IRIS FLOWER DATASET ANALYSIS")
    print("=" * 60)
    
    # Task 1: Loading and exploring the dataset
    df = load_and_explore_dataset()
    
    if df is not None:
        # Task 2: Basic data analysis
        species_group = basic_data_analysis(df)
        
        # Task 3: Data visualization
        create_visualizations(df, species_group)
        
        # Summary of findings
        print("\n" + "=" * 60)
        print("SUMMARY OF FINDINGS")
        print("=" * 60)
        print("1. The dataset contains 150 samples of 3 iris species with 4 measurements each.")
        print("2. Setosa species has distinctly smaller petals compared to other species.")
        print("3. Virginica species generally has the largest measurements.")
        print("4. Petal length and width show strong positive correlation.")
        print("5. The species are well-separated in measurement space, making classification feasible.")
        print("6. No missing values were found in the dataset.")
        
    else:
        print("Failed to load dataset. Please check the error message above.")

# Exception handling for the entire script
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user maybe.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
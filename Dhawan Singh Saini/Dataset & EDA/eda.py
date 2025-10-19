import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set global plotting style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

# --- 1. Data Loading and Initial Cleaning ---
file_path = 'enriched_crop_yield_2.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please ensure the file is in the correct directory.")
    exit()

# Standardize column names (lowercase, replace spaces, correct common typos)
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
df = df.rename(columns={'humpidity': 'humidity'})

# Convert Crop_Year to integer and ensure numerical columns are numeric
df['crop_year'] = df['crop_year'].astype(int)
numerical_cols = ['area', 'production', 'annual_rainfall', 'fertilizer', 'pesticide', 'humidity', 'avg_temperature', 'yield']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any NaN values in critical columns for cleaner visualizations
df_cleaned = df.dropna(subset=['crop', 'state', 'crop_year'] + numerical_cols)
print(f"Data after dropping NA in critical columns: {df_cleaned.shape}")

# Clean string columns (strip whitespace)
for col in ['crop', 'season', 'state', 'soil_type']:
    df_cleaned[col] = df_cleaned[col].str.strip()

# --- 2. Initial Required Distribution Plots (3 Graphs) ---

# Graph 1: Different types of crop in database (Top 20)
plt.figure(figsize=(14, 7))
top_crops = df_cleaned['crop'].value_counts().nlargest(20)
sns.barplot(x=top_crops.index, y=top_crops.values, palette="viridis")
plt.title('Distribution of Different Crop Types (Top 20)', fontweight='bold')
plt.xlabel('Crop Type')
plt.ylabel('Number of Records')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Graph 2: Crop distribution year wise
plt.figure(figsize=(12, 6))
sns.histplot(df_cleaned['crop_year'], bins=sorted(df_cleaned['crop_year'].unique()), kde=False, color='skyblue')
plt.title('Crop Distribution Year Wise (Number of Records per Year)', fontweight='bold')
plt.xlabel('Crop Year')
plt.ylabel('Number of Records')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Graph 3: Crop distribution state wise (Top 15)
plt.figure(figsize=(12, 6))
top_states = df_cleaned['state'].value_counts().nlargest(15)
sns.barplot(x=top_states.index, y=top_states.values, palette="rocket")
plt.title('Crop Distribution State Wise (Top 15 States)', fontweight='bold')
plt.xlabel('State')
plt.ylabel('Number of Records')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- 3. Area vs Yield Analysis Over the Years (Generating 10 Plots) ---

# Aggregate data by year and crop (Sum Area, Mean Yield)
df_agg = df_cleaned.groupby(['crop_year', 'crop']).agg({
    'area': 'sum',
    'yield': 'mean'
}).reset_index()

# Get the top 10 crops by total recorded area for diversified plotting
top_10_crops = df_cleaned.groupby('crop')['area'].sum().nlargest(10).index.tolist()

def plot_area_yield_trend(df_plot, crop_name, ax):
    """Generates a combined Area vs. Yield plot over the years."""
    if df_plot.empty:
        ax.text(0.5, 0.5, f"No data for {crop_name}", transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f"Area & Yield Trend for {crop_name} (No Data)")
        return

    # Area Plot (Left Y-axis)
    color_area = 'tab:blue'
    ax.plot(df_plot['crop_year'], df_plot['area'], marker='o', color=color_area, label='Total Area (Sum)')
    ax.set_xlabel('Crop Year')
    ax.set_ylabel('Total Area (in Hectares)', color=color_area)
    ax.tick_params(axis='y', labelcolor=color_area)

    # Yield Plot (Right Y-axis)
    ax2 = ax.twinx()
    color_yield = 'tab:red'
    ax2.plot(df_plot['crop_year'], df_plot['yield'], marker='^', linestyle='--', color=color_yield, label='Average Yield (Mean)')
    ax2.set_ylabel('Average Yield (Units)', color=color_yield)
    ax2.tick_params(axis='y', labelcolor=color_yield)

    ax.set_title(f'Area & Yield Trend Over Years for {crop_name}', fontweight='bold')
    ax.grid(True, linestyle='--')
    ax.set_xticks(df_plot['crop_year'].unique()) # Ensure all years are marked
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()


# Generate plots for the top 10 crops (Total 10 plots)
print(f"\nGenerating Area vs. Yield Trend Plots for Top {len(top_10_crops)} Crops...")
for i, crop in enumerate(top_10_crops):
    df_crop = df_agg[df_agg['crop'] == crop]

    # Skip if filtered data is empty
    if df_crop.empty:
        print(f"Skipping {crop}: Insufficient data for time series analysis.")
        continue

    # Plot specific requests first (Rice, Wheat)
    if crop == 'Rice':
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_area_yield_trend(df_crop, 'Rice', ax)
        plt.show() # Graph 4: Rice
    elif crop == 'Wheat':
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_area_yield_trend(df_crop, 'Wheat', ax)
        plt.show() # Graph 5: Wheat
    else:
        # Plotting the 'Other' major crops
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_area_yield_trend(df_crop, crop, ax)
        plt.show() # Graphs 6-13: Other Top Crops

# --- 4. Yield vs Environmental Factors (Generating 12 Plots) ---

# Select the top 3 crops for in-depth environmental correlation analysis
crops_for_analysis = top_10_crops[:3]
environmental_factors = ['annual_rainfall', 'fertilizer', 'pesticide', 'avg_temperature']

print(f"\nGenerating Scatter Plots of Yield vs. Environmental Factors for: {', '.join(crops_for_analysis)}...")
plot_counter = 14 # Start counter for the total 30+ graphs

for crop in crops_for_analysis:
    df_crop = df_cleaned[df_cleaned['crop'] == crop]
    if df_crop.empty: continue

    for factor in environmental_factors:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=factor, y='yield', data=df_crop, alpha=0.6, hue='state', palette='viridis')
        plt.title(f'Yield vs. {factor.replace("_", " ").title()} for {crop}', fontweight='bold')
        plt.xlabel(factor.replace("_", " ").title())
        plt.ylabel('Yield')
        plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.show()
        plot_counter += 1 # Graphs 14-25 (4 factors * 3 crops = 12 plots)


# --- 5. Additional Diverse EDA Plots (Generating 5+ Plots) ---

# Graph 26: Average Yield by Soil Type
plt.figure(figsize=(12, 6))
soil_yield = df_cleaned.groupby('soil_type')['yield'].mean().sort_values(ascending=False).nlargest(10)
sns.barplot(x=soil_yield.index, y=soil_yield.values, palette="tab10")
plt.title('Average Crop Yield by Soil Type (Top 10)', fontweight='bold')
plt.xlabel('Soil Type')
plt.ylabel('Average Yield')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Graphs 27-30: Distribution of key factors across states (Box Plots)
print("\nGenerating Box Plots for Environmental Factor Distribution across States...")
factors_for_boxplots = ['annual_rainfall', 'fertilizer', 'pesticide', 'avg_temperature']
top_10_states = df_cleaned['state'].value_counts().nlargest(10).index.tolist()
df_box = df_cleaned[df_cleaned['state'].isin(top_10_states)]

for factor in factors_for_boxplots:
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='state', y=factor, data=df_box, palette="Pastel1")
    plt.title(f'Distribution of {factor.replace("_", " ").title()} across Top 10 States', fontweight='bold')
    plt.xlabel('State')
    plt.ylabel(factor.replace("_", " ").title())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show() # Graphs 27-30 (4 plots)


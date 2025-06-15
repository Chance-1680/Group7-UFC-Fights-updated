import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re
import os

# Set seaborn style
sns.set_style("darkgrid")

# Create output directory for plots if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Load UFC cleaned dataset
data_path = r"C:\Users\USER\Desktop\Group7-UFC-Fights\data\processed\UFC_Cleaned Dataset1.csv"
df = pd.read_csv(data_path, low_memory=False)

#////////////////////////////////////////////////////////////////////
# Drop columns with mostly null values (less than 80% non-null)
df_cleaned = df.dropna(axis=1, thresh=len(df) * 0.8)

# Preview dataset
df_preview = df_cleaned

# Build scrollable table using Plotly
fig = go.Figure(data=[go.Table(
    header=dict(
        values=[f"<b>{col}</b>" for col in df_preview.columns],
        fill_color='lightblue',
        align='left',
        font=dict(size=12)
    ),
    cells=dict(
        values=[df_preview[col].astype(str).tolist() for col in df_preview.columns],
        fill_color='lavender',
        align='left',
        font=dict(size=11),
        height=25
    )
)])

# Set fixed height to enable vertical scroll inside the Plotly viewer (in browsers)
fig.update_layout(
    width=4000,  # You can adjust width as needed
    height=800,  # Table height to enable scroll if needed
    margin=dict(t=10, b=10)
)

# Show the table (will open in browser or VS Code plot viewer)
fig.show()

# Save the table as a standalone HTML file for later viewing/sharing
fig.write_html("scrollable_table.html")
print("Scrollable table saved to scrollable_table.html")
#////////////////////////////////////////////////////////////////////



# Drop columns with mostly null values (less than 80% non-null)
df_cleaned = df.dropna(axis=1, thresh=len(df)*0.8)

# Print dataset summary
print(f"Dataset Summary:\n- Rows: {df.shape[0]}\n- Features: {df.shape[1]}")
print("\nColumns in dataset:", df.columns.tolist())
print("\nPreview of cleaned data:\n", df_cleaned.head(10).to_string())

# 1. Distribution of Knockdowns
df['Knockdowns'] = pd.to_numeric(df['Knockdowns'], errors='coerce')
max_knockdowns = int(df['Knockdowns'].max()) if not df['Knockdowns'].isna().all() else 0

plt.figure(figsize=(8, 5))
ax = sns.histplot(data=df, x='Knockdowns', bins=range(0, max_knockdowns + 2), discrete=True, kde=True)
plt.title('Knockdowns per UFC Fight (Distribution)', fontsize=14)
plt.xlabel('Number of Knockdowns', fontsize=12)
plt.ylabel('Number of Fights', fontsize=12)
plt.xlim(0, 3)

# Add value labels on bars
for patch in ax.patches:
    height = patch.get_height()
    if height > 0:
        ax.annotate(f'{int(height):,}',
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/knockdowns_distribution.png")
plt.close()

# 2. Extract Strikes Data
def safe_extract_strikes(strikes_str):
    if pd.isna(strikes_str) or str(strikes_str) == '--':
        return pd.Series([None, None])
    match = re.search(r'(\d+).*?(\d+)', str(strikes_str))
    if match:
        return pd.Series([int(match.group(1)), int(match.group(2))])
    return pd.Series([None, None])

df[['Strikes_Landed', 'Strikes_Attempted']] = df['Strikes'].apply(safe_extract_strikes)
print("\nSample of extracted strikes data:\n", df[['Fighter', 'Strikes', 'Strikes_Landed', 'Strikes_Attempted']].head(10).to_string())

successful = df['Strikes_Landed'].notna().sum()
print(f"\nSuccessful extractions: {successful} out of {len(df)}")
if successful > 0:
    print(f"Strikes_Landed range: {df['Strikes_Landed'].min()} to {df['Strikes_Landed'].max()}")

# Top 10 Fighters by Total Strikes Landed
top_strikers = df.groupby('Fighter')['Strikes_Landed'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Fighters by Total Strikes Landed:\n", top_strikers.to_string())

plt.figure(figsize=(8, 5))
sns.barplot(x=top_strikers.values, y=top_strikers.index, color='#1f77b4', orient='h')
plt.title('Top 10 Fighters by Total Strikes Landed', fontsize=12, fontweight='bold')
plt.xlabel('Total Strikes Landed', fontsize=10)
plt.ylabel('Fighter', fontsize=10)

for i, v in enumerate(top_strikers.values):
    plt.text(v + max(top_strikers.values) * 0.01, i, f'{int(v)}', 
             va='center', fontweight='bold')

plt.tight_layout()
plt.savefig("plots/top_strikers.png")
plt.close()

# 3. Average Fight Length by Weight Class
df['Weight_Clean'] = df['Weight'].str.extract(r'(\d+)').astype(float)
weight_class_map = {
    115: 'Strawweight', 125: 'Flyweight', 135: 'Bantamweight',
    145: 'Featherweight', 155: 'Lightweight', 170: 'Welterweight',
    185: 'Middleweight', 205: 'Light Heavyweight', 265: 'Heavyweight'
}
df['Weight_Class'] = df['Weight_Clean'].map(weight_class_map)
df = df.dropna(subset=['Weight_Class'])

df = df[df['Round'].notna() & (df['Round'] > 0)].copy()
df['Fight_Seconds'] = df['Round'] * 300
avg_fight_length = df.groupby('Weight_Class')['Fight_Seconds'].mean().sort_values()

plt.figure(figsize=(8, 5))
sns.barplot(x=avg_fight_length.values, y=avg_fight_length.index, color='#1f77b4')
plt.title('Average Fight Length by Weight Class', fontsize=14, fontweight='bold')
plt.xlabel('Average Fight Length (seconds)', fontsize=11)
plt.ylabel('Weight Class', fontsize=11)

for i, v in enumerate(avg_fight_length.values):
    plt.text(v + 5, i, f'{int(v)}s', va='center')

plt.tight_layout()
plt.savefig("plots/avg_fight_length.png")
plt.close()

# 4. Win Method Trends
def simplify_method(method):
    method = str(method).upper()
    if 'KO' in method:
        return 'KO/TKO'
    elif 'SUB' in method:
        return 'Submission'
    elif 'DEC' in method:
        return 'Decision'
    elif 'DQ' in method:
        return 'Disqualification'
    elif 'OVERTURNED' in method:
        return 'Overturned'
    else:
        return 'Other'

df['Win_Method_Simple'] = df['Method'].apply(simplify_method)
method_counts = df['Win_Method_Simple'].value_counts()
print("\nAvailable win methods:", method_counts.index.tolist())

custom_order = ['Decision', 'Other', 'Submission', 'Overturned', 'KO/TKO', 'Disqualification']
method_counts = method_counts[custom_order]

plt.figure(figsize=(8, 5))
colors = sns.color_palette('Set2', n_colors=len(method_counts))
method_counts.plot.pie(
    autopct='%1.1f%%',
    startangle=100,
    colors=colors,
    textprops={'fontsize': 12},
    labeldistance=1.05
)
plt.ylabel('')
plt.title('Distribution of Win Methods', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/win_methods.png")
plt.close()

# 5. Fight Frequency Over Time
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_clean = df[df['Date'].notna()].copy()
fights_per_year_count = df_clean[df_clean['Date'].dt.year >= 1998]['Date'].dt.year.value_counts().sort_index()
fights_per_year_df = fights_per_year_count.reset_index()
fights_per_year_df.columns = ['Year', 'Fights']

plt.figure(figsize=(10, 5))
sns.lineplot(data=fights_per_year_df, x='Year', y='Fights', marker='o', linewidth=2)
plt.title('Fight Frequency Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Fights', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/fight_frequency.png")
plt.close()

print("\nAnalysis complete. Plots saved in 'plots' directory.")

print(df.head())  # or just df.head()





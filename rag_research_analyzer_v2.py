import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA
filename = "squad_research_results.csv"
print(f"📂 Loading research data from {filename}...")

try:
    df = pd.read_csv(filename)
    df = df.dropna(subset=['context_precision', 'faithfulness', 'answer_relevancy'])
except FileNotFoundError:
    print(f"❌ File '{filename}' not found!")
    exit()

# 2. GENERATE CLEANER PLOTS
print("🎨 Generating Professional Plots...")

# Melt data for easier plotting
df_melted = df.melt(id_vars=['method'], 
                    value_vars=['context_precision', 'faithfulness', 'answer_relevancy'],
                    var_name='Metric', value_name='Score')

# --- PLOT 1: The "Split" Box Plot (Professional Standard) ---
# We create 3 separate mini-graphs arranged vertically.

g = sns.FacetGrid(df_melted, col="Metric", col_wrap=3, height=5, aspect=0.8, sharey=True)

# FIX 1: Removed 'alpha' (caused the crash)
# FIX 2: Added 'hue' and 'legend=False' (fixes the warning)
g.map_dataframe(sns.boxplot, x="method", y="Score", hue="method", palette="viridis", legend=False)

# Overlay actual data points (Strip Plot)
# alpha works fine here
g.map_dataframe(sns.stripplot, x="method", y="Score", color="black", size=3, alpha=0.4, jitter=True)

# Clean up axes
g.set_axis_labels("", "Score (0-1)")
g.set_titles(col_template="{col_name}")

# Rotate x-axis labels for readability
for axes in g.axes.flat:
    _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')

plt.suptitle("RAG Research: Metric Distributions (Points + Box)", y=1.05, fontsize=16)
plt.show()

# --- PLOT 2: The Main Comparison (Bar Chart) ---
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
sns.barplot(data=df_melted, x='method', y='Score', hue='Metric', palette="viridis", errorbar=None)
plt.title("RAG Research: Average Performance", fontsize=16)
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("assets/performance_comparison.jpg")
print("💾 Saved plot to assets/performance_comparison.jpg")
plt.close()

print("✅ Analysis Complete.")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog, Tk
from glob import glob
from scipy.stats import shapiro, normaltest, anderson, ttest_ind, mannwhitneyu, sem, normaltest

# --- File selection ---
root = Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Select folder with *_nuc_cyto_area_ratio_FINAL.csv files")
if not folder_path:
    raise SystemExit("No folder selected.")

files = sorted(glob(os.path.join(folder_path, "*_nuc_cyto_area_ratio_FINAL.csv")))
if not files:
    raise FileNotFoundError("No matching CSV files found in the selected folder.")

# --- Organize and merge data ---
all_data = []
group_labels = []

for file in files:
    df = pd.read_csv(file)
    label = "AX4" if "AX4" in os.path.basename(file) else "DnmA"
    df["Source_File"] = os.path.basename(file)
    df["Group"] = label
    all_data.append(df)

master_df = pd.concat(all_data, ignore_index=True)

# --- Save full master dataframe ---
master_output_path = os.path.join(folder_path, "Master_Cell_Ratios_Combined.csv")
master_df.to_csv(master_output_path, index=False)
print(f"[✔] Master CSV saved to: {master_output_path}")

# --- Extract Area Ratios by Group ---
ax4_ratios = master_df[master_df["Group"] == "AX4"]["Area_Ratio"].dropna()
dnma_ratios = master_df[master_df["Group"] == "DnmA"]["Area_Ratio"].dropna()

# --- Normality testing ---
def test_normality(data, label):
    results = {}
    results['Shapiro'] = shapiro(data)
    results['DAgostino'] = normaltest(data)
    results['Anderson'] = anderson(data)
    log_data = np.log(data[data > 0])
    results['Log_Shapiro'] = shapiro(log_data)
    results['Log_DAgostino'] = normaltest(log_data)
    results['log_normal'] = (results['Log_Shapiro'].pvalue > 0.05 or results['Log_DAgostino'].pvalue > 0.05)
    results['is_normal'] = (results['Shapiro'].pvalue > 0.05 or results['DAgostino'].pvalue > 0.05)
    print(f"\n--- Normality for {label} ---")
    for test, res in results.items():
        if test in ['Anderson']:
            print(f"{test}: statistic={res.statistic:.4f}, critical_values={res.critical_values}, significance={res.significance_level}")
        elif 'pvalue' in dir(res):
            print(f"{test}: statistic={res.statistic:.4f}, p={res.pvalue:.4f}")
    return results

ax4_norm = test_normality(ax4_ratios, "AX4")
dnma_norm = test_normality(dnma_ratios, "DnmA")

# --- Decide statistical test ---
def is_normal(norm_result):
    return norm_result['is_normal']

if is_normal(ax4_norm) and is_normal(dnma_norm):
    stat_test_name = "Unpaired t-test (Parametric)"
    stat_result = ttest_ind(ax4_ratios, dnma_ratios, equal_var=False)
else:
    stat_test_name = "Mann-Whitney U (Non-parametric)"
    stat_result = mannwhitneyu(ax4_ratios, dnma_ratios, alternative='two-sided')

# --- Compute group stats ---
def group_stats(data):
    return {
        "Mean": np.mean(data),
        "StdDev": np.std(data, ddof=1),
        "95% CI": sem(data) * 1.96,
        "Min": np.min(data),
        "Max": np.max(data),
        "N": len(data)
    }

ax4_stats = group_stats(ax4_ratios)
dnma_stats = group_stats(dnma_ratios)

# --- Create Stats Summary ---
stats_summary = {
    "AX4 Count": ax4_stats["N"],
    "DnmA Count": dnma_stats["N"],
    "Test Used": stat_test_name,
    "Statistic": stat_result.statistic,
    "p-value": stat_result.pvalue,
    "Significant (p<0.05)": stat_result.pvalue < 0.05,
    "AX4 Normal?": ax4_norm['is_normal'],
    "AX4 Log-Normal?": ax4_norm['log_normal'],
    "DnmA Normal?": dnma_norm['is_normal'],
    "DnmA Log-Normal?": dnma_norm['log_normal'],
    "AX4 Shapiro p": ax4_norm['Shapiro'].pvalue,
    "DnmA Shapiro p": dnma_norm['Shapiro'].pvalue,
    "AX4 DAgostino p": ax4_norm['DAgostino'].pvalue,
    "DnmA DAgostino p": dnma_norm['DAgostino'].pvalue,
    "AX4 Mean ± SD": f"{ax4_stats['Mean']:.3f} ± {ax4_stats['StdDev']:.3f}",
    "AX4 95% CI": ax4_stats['95% CI'],
    "AX4 Range": f"{ax4_stats['Min']:.3f} - {ax4_stats['Max']:.3f}",
    "DnmA Mean ± SD": f"{dnma_stats['Mean']:.3f} ± {dnma_stats['StdDev']:.3f}",
    "DnmA 95% CI": dnma_stats['95% CI'],
    "DnmA Range": f"{dnma_stats['Min']:.3f} - {dnma_stats['Max']:.3f}"
}

summary_df = pd.DataFrame([stats_summary])
summary_path = os.path.join(folder_path, "Stats_Summary_AX4_vs_DnmA.csv")
summary_df.to_csv(summary_path, index=False)
print(f"[✔] Statistics summary saved: {summary_path}")

# --- Histogram of Area Ratios (Combined) ---
plt.figure(figsize=(8, 5))
sns.histplot(data=master_df, x="Area_Ratio", hue="Group", kde=True, bins=30, palette="muted")
plt.title("Distribution of Area Ratios")
plt.xlabel("Nucleus / Cytoplasm Area Ratio")
plt.ylabel("Frequency")
histogram_path_svg = os.path.join(folder_path, "histogram_area_ratios.svg")
histogram_path_png = os.path.join(folder_path, "histogram_area_ratios.png")
plt.savefig(histogram_path_svg, format='svg', dpi=300)
plt.savefig(histogram_path_png, format='png', dpi=300)
plt.show()

# --- Individual histograms ---
for group, data in master_df.groupby("Group"):
    plt.figure(figsize=(6, 4))
    sns.histplot(data["Area_Ratio"], kde=True, bins=30, color="steelblue")
    plt.title(f"Histogram of Area Ratios: {group}")
    plt.xlabel("Nucleus / Cytoplasm Area Ratio")
    plt.ylabel("Frequency")
    group_path_svg = os.path.join(folder_path, f"histogram_area_ratios_{group}.svg")
    group_path_png = os.path.join(folder_path, f"histogram_area_ratios_{group}.png")
    plt.savefig(group_path_svg, format='svg', dpi=300)
    plt.savefig(group_path_png, format='png', dpi=300)
    plt.show()

# --- Bar Plot with individual points ---
plt.figure(figsize=(6, 5))
sns.boxplot(data=master_df, x="Group", y="Area_Ratio", showfliers=False, palette="pastel")
sns.stripplot(data=master_df, x="Group", y="Area_Ratio", jitter=True, alpha=0.6, color="black")
plt.title("Area Ratio by Group")
plt.ylabel("Nucleus / Cytoplasm Area Ratio")

# Annotate with stats
x_center = 0.5
y_max = master_df["Area_Ratio"].max() * 1.1
p_value = stat_result.pvalue
plt.text(x_center, y_max, f"{stat_test_name}\nP = {p_value:.3e}{' *' if p_value < 0.05 else ''}",
         ha='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

boxplot_path_svg = os.path.join(folder_path, "boxplot_with_stats.svg")
boxplot_path_png = os.path.join(folder_path, "boxplot_with_stats.png")
plt.savefig(boxplot_path_svg, format='svg', dpi=300)
plt.savefig(boxplot_path_png, format='png', dpi=300)
plt.show()

print("\n[✔] All plots and outputs successfully saved.")

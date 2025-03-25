# %%
import pandas as pd

# %%
description = pd.read_csv("./WiDS_Datathon_2020_Dictionary.csv")
description_dict = description.set_index("Variable Name").to_dict(orient="index")

df = pd.read_csv("./training_v2.csv")

df.head()

# %%
description_dict["h1_spo2_min"]

# %%
description_dict["d1_bilirubin_max"]

# %%
description_dict["d1_bun_max"]

# %%
description_dict["h1_hematocrit_max"]

# %%
len(df.columns)
"""
There are 186 columns in the dataset
"""

# %%
per_missing: pd.Series = (df.isnull().sum() / df.shape[0] * 100).sort_values(
    ascending=False
)
per_missing[per_missing > 50]
len(per_missing[per_missing > 50])
"""
74 columns have more than 50% missing values
"""

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
summary_stats = df.describe()
print(summary_stats)

# %%
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
for col in numerical_columns[:5]:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# %%
correlation_matrix = df.apply(
    lambda x: pd.factorize(x)[0] if x.dtype == "object" else x
).corr(method="pearson")
plt.figure(figsize=(26, 20))
sns.heatmap(
    correlation_matrix, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True
)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.title("Correlation Matrix", fontsize=16)
plt.savefig("correlation_matrix.png", dpi=600)
plt.show()

# %%
import numpy as np

correlation_matrix = df.apply(
    lambda x: pd.factorize(x)[0] if x.dtype == "object" else x
).corr(method="pearson")

# Set a correlation threshold to identify highly correlated feature pairs
threshold = 0.8  # Adjust based on your needs

# Create a mask for the upper triangle of the correlation matrix (to avoid duplicates)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Find pairs of features with correlation above threshold
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) >= threshold:
            high_corr_pairs.append(
                (
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j],
                )
            )

# Sort by absolute correlation value (descending)
high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# Get unique features that appear in these high correlation pairs
high_corr_features = set()
for feat1, feat2, _ in high_corr_pairs:
    high_corr_features.add(feat1)
    high_corr_features.add(feat2)

# Convert to list and sort for consistency
high_corr_features = sorted(list(high_corr_features))

# Create a smaller correlation matrix with just these features
smaller_corr_matrix = correlation_matrix.loc[high_corr_features, high_corr_features]

# Plot the smaller correlation matrix
plt.figure(figsize=(18, 16))
sns.heatmap(
    smaller_corr_matrix,
    annot=False,  # Set to True if the matrix is small enough
    cmap="coolwarm",
    xticklabels=True,
    yticklabels=True,
    vmin=-1,
    vmax=1,
)
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8)
plt.title(
    f"Correlation Matrix of Highly Correlated Features (|r| ≥ {threshold})", fontsize=16
)
plt.savefig("high_correlation_matrix.pdf", dpi=300, format="pdf", bbox_inches="tight")
plt.show()

# Optionally, print out the highly correlated pairs for reference
print(f"Top {min(20, len(high_corr_pairs))} highly correlated feature pairs:")
for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:20], 1):
    print(f"{i}. {feat1} — {feat2}: {corr:.3f}")

# %%
hospital_death_correlation = (
    df.apply(lambda x: pd.factorize(x)[0] if x.dtype == "object" else x)
    .corrwith(df["hospital_death"])
    .sort_values(ascending=False)
)
hospital_death_correlation.head(10)
hospital_death_correlation.tail(10)
"""
d1_lactate_min                   0.403614
d1_lactate_max                   0.399029
h1_lactate_min                   0.344046
h1_lactate_max                   0.340951
apache_4a_hospital_death_prob    0.311043
apache_4a_icu_death_prob         0.283913
ventilated_apache                0.228661
fio2_apache                      0.212249
h1_inr_max                       0.198641

d1_sysbp_min            -0.210170
d1_mbp_invasive_min     -0.222350
h1_albumin_max          -0.224928
h1_albumin_min          -0.225402
d1_arterial_ph_min      -0.230365
d1_sysbp_invasive_min   -0.234382
gcs_verbal_apache       -0.241044
gcs_eyes_apache         -0.260373
gcs_motor_apache        -0.282449
"""

# %%
description_dict["d1_lactate_min"]
description_dict["h1_lactate_min"]
description_dict["apache_4a_hospital_death_prob"]

description_dict["gcs_motor_apache"]
description_dict["gcs_eyes_apache"]

# %%
"""
lactate levels are positively correlated with hospital death
lactate is produced by the body when there is not enough oxygen in the tissues; this means if one cannot breath/ get enough oxygen
"""

# %%
df["d1_lactate_min"].describe()
df["d1_lactate_min"].isnull().sum() / df.shape[0] * 100
"""
High missing values with nearly 75% of the values missing
"""

# %%
df["d1_lactate_min"].fillna(0).corr(df["hospital_death"])  # type: ignore
"""
This lowers correlation to ~32% from ~40%. This means that the missing values can not be described as just a zero value.
"""

df["d1_lactate_min"].fillna(df["d1_lactate_min"].mean()).corr(df["hospital_death"])  # type: ignore
"""
Even worse.
"""

df["d1_lactate_min"].fillna(df["d1_lactate_min"].min()).corr(df["hospital_death"])  # type: ignore
"""
Around the same as 0 as the min is 0.4 anyway
"""

df["d1_lactate_min"].fillna(df["d1_lactate_min"].max()).corr(df["hospital_death"])  # type: ignore
"""
Really bad. Changes the whole correlation to negative
"""

# %%
df["h1_lactate_min"].describe()
df["h1_lactate_min"].isnull().sum() / df.shape[0] * 100
"""
Even higher missing values with over 90% of the values missing
"""

df["h1_lactate_min"].fillna(0).corr(df["hospital_death"])  # type: ignore
"""
This lowers correlation to ~16% from ~32%. This means that the missing values can not be described as just a zero value.
"""

# %%
df["gcs_motor_apache"].describe()
df["gcs_motor_apache"].isnull().sum() / df.shape[0] * 100
"""
Only ~2% of the values are missing
"""

df["gcs_motor_apache"].value_counts()
"""
The value are categorical. The values are 1, 2, 3, 4, 5, 6. 6 means they obey cmd, 5 means they obey verbal, 4 means they obey pain, 3 means they have flexion, 2 means they have extension, 1 means they have no response
This means the higher the value, the better the response
A negative correlation therfore makes sense as the lower the value the more likely they are to die as they cannot control motoric functions
The variable is ordinal and therefore spearman/kendall should be used instead of pearson (I think)
Most of the rows has a value of 6
"""

# %%
df["gcs_motor_apache"].corr(df["hospital_death"], method="spearman")  # type: ignore
df["gcs_motor_apache"].corr(df["hospital_death"], method="kendall")  # type: ignore


# %%
categorical_columns = df.select_dtypes(include=["object"]).columns
unique_values = {col: df[col].nunique() for col in categorical_columns}
print("Unique values in categorical columns:", unique_values)

# %%
for col in numerical_columns[:5]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Box plot of {col}")
    plt.show()

# %%
data_types = df.dtypes
print("Data types of each column:", data_types)

# %%
df_dropped_cols = df.drop(columns=[col for col in df.columns if "apache" in col])

# %%
correlation_matrix = df_dropped_cols.apply(
    lambda x: pd.factorize(x)[0] if x.dtype == "object" else x
).corr(method="pearson")
plt.figure(figsize=(26, 20))
sns.heatmap(
    correlation_matrix, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True
)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.title("Correlation Matrix", fontsize=16)
plt.savefig("correlation_matrix.pdf", dpi=600, format="pdf", bbox_inches="tight")
plt.show()

# %%
df["gender"].value_counts()

# %%
df[df["gender"] == "M"]["hospital_death"].value_counts()

# %%
df[df["gender"] == "F"]["hospital_death"].value_counts()

# %%
df["ethnicity"].value_counts()

# %%
df[df["ethnicity"] == "Caucasian"]["hospital_death"].value_counts()

# %%
df[df["ethnicity"] == "Native American"]["hospital_death"].value_counts()

# %%
df[df["ethnicity"] == "African American"]["hospital_death"].value_counts()

# Calculate the total number of deaths and total records for each gender
gender_death_counts = df[df["hospital_death"] == 1]["gender"].value_counts()
gender_total_counts = df["gender"].value_counts()

# Calculate the percentage of deaths per gender
gender_death_percentage = (gender_death_counts / gender_total_counts) * 100

# Plot the bar chart
fig, ax = plt.subplots(figsize=(8, 5))
gender_death_percentage.plot(kind="bar", color="lightcoral", ax=ax)
ax.set_title("Percentage of Deaths per Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Percentage of Deaths")
fig.savefig("gender_precentage_deaths.pdf", dpi=600, format="pdf", bbox_inches="tight")

# %%
# Calculate the total number of deaths and total records for each ethnicity
ethnicity_death_counts = df[df["hospital_death"] == 1]["ethnicity"].value_counts()
ethnicity_total_counts = df["ethnicity"].value_counts()

# Calculate the percentage of deaths per ethnicity
ethnicity_death_percentage = (ethnicity_death_counts / ethnicity_total_counts) * 100

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ethnicity_death_percentage.plot(kind="bar", color="skyblue")
ax.set_title("Percentage of Deaths per Ethnicity")
ax.set_xlabel("Ethnicity")
ax.set_ylabel("Percentage of Deaths")
fig.savefig(
    "ethnicity_precentage_deaths.pdf", dpi=600, format="pdf", bbox_inches="tight"
)

# %%
# Calculate the total number of deaths and total records for each ethnicity
ethnicity_death_counts = df[df["hospital_death"] == 1]["ethnicity"].value_counts()
ethnicity_total_counts = df["ethnicity"].value_counts()

# Calculate the percentage of deaths per ethnicity
ethnicity_death_percentage = (ethnicity_death_counts / ethnicity_total_counts) * 100

# Sort the data in descending order for better visualization
ethnicity_death_percentage = ethnicity_death_percentage.sort_values(ascending=False)

# Set a clean, modern style
# plt.style.use("seaborn-v0_8-whitegrid")
plt.style.use("default")

# Create the figure with improved dimensions
fig, ax = plt.subplots(figsize=(12, 8))

# Create the bar chart with a better color palette and alpha for depth
bars = ethnicity_death_percentage.plot(
    kind="bar",
    # color=plt.cm.viridis(np.linspace(0.1, 0.9, len(ethnicity_death_percentage))),
    alpha=0.8,
    ax=ax,
)

# Enhance the title and labels with better fonts and styling
# ax.set_title("Mortality Rate by Ethnicity", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Ethnicity", fontsize=14, labelpad=10)
ax.set_ylabel("Mortality Rate (%)", fontsize=14, labelpad=10)

# Add data labels on top of each bar
for bar in bars.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Add a subtle grid on the y-axis only for better readability
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Remove top and right spines for cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Rotate the x-axis labels slightly and align them
plt.xticks(rotation=30, ha="right", fontsize=12)
plt.yticks(fontsize=12)

# Adjust layout to make sure everything fits nicely
plt.tight_layout()

# Save with higher quality
fig.savefig(
    "ethnicity_percentage_deaths.pdf", dpi=600, format="pdf", bbox_inches="tight"
)

# Also save as PNG for quick viewing
fig.savefig(
    "ethnicity_percentage_deaths.png", dpi=300, format="png", bbox_inches="tight"
)

# Display the plot
plt.show()

# %%
df["age"].describe()

# Calculate the total number of deaths and total records for each age
age_death_counts = df[df["hospital_death"] == 1]["age"].value_counts().sort_index()
age_total_counts = df["age"].value_counts().sort_index()

# Calculate the percentage of deaths per age
age_death_percentage = (age_death_counts / age_total_counts) * 100

# Plot the line chart
plt.figure(figsize=(12, 6))
age_death_percentage.plot(kind="line", marker="o", color="green")
plt.title("Percentage of Deaths per Age")
plt.xlabel("Age")
plt.ylabel("Percentage of Deaths")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
list(df.columns)

# %%
df["hospital_admit_source"].value_counts()


# %%
gender_death_counts = df[df["hospital_death"] == 1]["gender"].value_counts()
gender_total_counts = df["gender"].value_counts()

# Calculate the percentage of deaths per gender
gender_death_percentage = (gender_death_counts / gender_total_counts) * 100

# Sort the data in descending order for better visualization
gender_death_percentgender = gender_death_percentage.sort_values(ascending=False)

# Set a clean, modern style
# plt.style.use("seaborn-v0_8-whitegrid")
plt.style.use("default")

# Create the figure with improved dimensions
fig, ax = plt.subplots(figsize=(8, 8))

# Create the bar chart with a better color palette and alpha for depth
bars = gender_death_percentgender.plot(
    kind="bar",
    # color=plt.cm.viridis(np.linspace(0.1, 0.9, len(gender_death_percentgender))),
    alpha=0.8,
    ax=ax,
)

# Enhance the title and labels with better fonts and styling
# ax.set_title("Mortality Rate by gender", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("gender", fontsize=14, labelpad=10)
ax.set_ylabel("Mortality Rate (%)", fontsize=14, labelpad=10)

# Add data labels on top of each bar
for bar in bars.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Add a subtle grid on the y-axis only for better readability
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Remove top and right spines for cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Rotate the x-axis labels slightly and align them
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

# Adjust layout to make sure everything fits nicely
plt.tight_layout()

# Save with higher quality
fig.savefig("gender_percentage_deaths.pdf", dpi=600, format="pdf", bbox_inches="tight")

# Also save as PNG for quick viewing
fig.savefig("gender_percentage_deaths.png", dpi=300, format="png", bbox_inches="tight")

# %%

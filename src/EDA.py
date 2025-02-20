# %%
import pandas as pd

# %%
description = pd.read_csv(
    "./physionet.org/files/widsdatathon2020/1.0.0/data/WiDS_Datathon_2020_Dictionary.csv"
)
description_dict = description.set_index("Variable Name").to_dict(orient="index")

df = pd.read_csv("./physionet.org/files/widsdatathon2020/1.0.0/data/training_v2.csv")

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
df["d1_lactate_min"].fillna(0).corr(df["hospital_death"]) # type: ignore
"""
This lowers correlation to ~32% from ~40%. This means that the missing values can not be described as just a zero value.
"""

df["d1_lactate_min"].fillna(df["d1_lactate_min"].mean()).corr(df["hospital_death"]) # type: ignore
"""
Even worse.
"""

df["d1_lactate_min"].fillna(df["d1_lactate_min"].min()).corr(df["hospital_death"]) # type: ignore
"""
Around the same as 0 as the min is 0.4 anyway
"""

df["d1_lactate_min"].fillna(df["d1_lactate_min"].max()).corr(df["hospital_death"]) # type: ignore
"""
Really bad. Changes the whole correlation to negative
"""

# %%
df["h1_lactate_min"].describe()
df["h1_lactate_min"].isnull().sum() / df.shape[0] * 100
"""
Even higher missing values with over 90% of the values missing
"""

df["h1_lactate_min"].fillna(0).corr(df["hospital_death"]) # type: ignore
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
df["gcs_motor_apache"].corr(df["hospital_death"], method="spearman") # type: ignore
df["gcs_motor_apache"].corr(df["hospital_death"], method="kendall") # type: ignore


# %%


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

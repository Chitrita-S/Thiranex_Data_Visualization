import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

for column in df.columns:

    if df[column].isnull().sum() > 0:

        if df[column].dtype in ["int64", "float64"]:
            df[column] = df[column].fillna(df[column].median())

        else:
            df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE VALUES")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicate rows removed.")

else:

    print("No duplicate rows found.")

print("\n" + "=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)

numeric_columns = df.select_dtypes(include="number").columns

outlier_summary = {}

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    outlier_summary[column] = len(outliers)

    print(
        f"{column}: {len(outliers)} outliers"
    )

df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset saved as: cleaned_data.csv")

sns.set_theme(style="whitegrid")

if "Salary" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["Salary"],
        bins=10,
        edgecolor="black"
    )

    plt.title("Salary Distribution")
    plt.xlabel("Salary")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("01_salary_distribution.png")

    plt.show()

if "Department" in df.columns and "Salary" in df.columns:

    department_salary = (
        df.groupby("Department")["Salary"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))

    department_salary.plot(kind="bar")

    plt.title("Average Salary by Department")
    plt.xlabel("Department")
    plt.ylabel("Average Salary")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig("02_average_salary_department.png")

    plt.show()

if "Department" in df.columns:

    department_count = df["Department"].value_counts()

    plt.figure(figsize=(7, 7))

    department_count.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Department Distribution")
    plt.ylabel("")

    plt.tight_layout()

    plt.savefig("03_department_distribution.png")

    plt.show()

if "Age" in df.columns and "Salary" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["Age"],
        df["Salary"],
        alpha=0.7
    )

    plt.title("Age vs Salary")
    plt.xlabel("Age")
    plt.ylabel("Salary")

    plt.tight_layout()

    plt.savefig("04_age_vs_salary.png")

    plt.show()

if "Salary" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df["Salary"]
    )

    plt.title("Salary Outlier Detection")
    plt.xlabel("Salary")

    plt.tight_layout()

    plt.savefig("05_salary_outliers.png")

    plt.show()

numeric_df = df.select_dtypes(include="number")

if len(numeric_df.columns) >= 2:

    plt.figure(figsize=(10, 7))

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("06_correlation_heatmap.png")

    plt.show()

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

if "Salary" in df.columns:

    highest_salary = df["Salary"].max()
    lowest_salary = df["Salary"].min()
    average_salary = df["Salary"].mean()

    print(f"\nAverage Salary : {average_salary:.2f}")
    print(f"Highest Salary : {highest_salary:.2f}")
    print(f"Lowest Salary  : {lowest_salary:.2f}")

if "Department" in df.columns:

    department_count = df["Department"].value_counts()

    print(
        f"\nMost populated department: "
        f"{department_count.idxmax()}"
    )

    if "Salary" in df.columns:

        department_salary = (
            df.groupby("Department")["Salary"]
            .mean()
        )

        print(
            f"Highest average salary department: "
            f"{department_salary.idxmax()}"
        )

        print(
            f"Highest average department salary: "
            f"{department_salary.max():.2f}"
        )

if "Age" in df.columns and "Salary" in df.columns:

    correlation_value = df["Age"].corr(df["Salary"])

    print(
        f"\nAge-Salary Correlation: "
        f"{correlation_value:.2f}"
    )

print("\n" + "=" * 60)
print("FINAL DATASET INFORMATION")
print("=" * 60)

print("\nFinal Shape:")
print(df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")
print("1. cleaned_data.csv")
print("2. 01_salary_distribution.png")
print("3. 02_average_salary_department.png")
print("4. 03_department_distribution.png")
print("5. 04_age_vs_salary.png")
print("6. 05_salary_outliers.png")
print("7. 06_correlation_heatmap.png")
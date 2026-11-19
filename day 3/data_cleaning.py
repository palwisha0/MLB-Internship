import pandas as pd

# 1. Dataset load karna
df = pd.read_csv("student_performance.csv")

# 2. Missing values check karna
print("Missing Values:")
print(df.isnull().sum())
df = df.fillna(df.mean(numeric_only=True))

# 3. Duplicate records hatana
df = df.drop_duplicates()

# 4. Column rename karna
df = df.rename(columns={"ML": "Machine_Learning"})

# 5. Average_Score column banana
df["Average_Score"] = (df["Math"] + df["Python"] + df["Machine_Learning"]) / 3

# 6. Performance column banana
def get_performance(avg):
    if avg >= 90:
        return "Excellent"
    elif avg >= 80:
        return "Good"
    elif avg >= 70:
        return "Average"
    else:
        return "Needs Improvement"

df["Performance"] = df["Average_Score"].apply(get_performance)

# 7. Cleaned file save karna
df.to_csv("cleaned_student_performance.csv", index=False)
print("Cleaned file ban gayi!")
print(df)

import pandas as pd


df = pd.read_csv("student_data.csv")
print("Original Data:")
print(df.head())
print("\nShape:", df.shape)


print("\nMissing Values:")
print(df.isnull().sum())


df = df.dropna()

df = df.drop_duplicates()
print("\nAfter Cleaning Shape:", df.shape)



df.to_csv("cleaned_student_data.csv", index=False)
print("\nCleaned data saved as cleaned_student_data.csv")
print(df.head())
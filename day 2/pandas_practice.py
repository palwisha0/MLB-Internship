import pandas as pd

data = {
    "Name": ["Ali", "Ayesha", "Bilal"],
    "Marks": [85, 92, 60]
}
df = pd.DataFrame(data)
print(df)
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())
print("\nAverage Marks:", df["Marks"].mean())
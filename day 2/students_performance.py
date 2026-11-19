import pandas as pd
df = pd.read_csv("students.csv")
print(df.head())
print(df.info())
print("Average per Subject:\n", df[["Math","English","Science"]].mean())
df["Total"] = df["Math"] + df["English"] + df["Science"]
df["Average"] = df["Total"]/3
print("\nTop 5:\n", df.sort_values(by="Total", ascending=False).head(5))
print("\nBelow Average:\n", df[df["Total"] < df["Total"].mean()])
print("\nTotal Students:", len(df))
df.to_csv("students_processed.csv", index=False)
print("Processed csv Saved!")
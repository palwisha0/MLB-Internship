import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_student_performance.csv")

# 1. Bar Chart
plt.figure()
plt.bar(df["Name"], df["Average_Score"])
plt.title("Average Marks of Each Student")
plt.xlabel("Student Name")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# 2. Histogram
plt.figure()
plt.hist(df["Average_Score"], bins=5)
plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# 3. Scatter Plot
plt.figure()
plt.scatter(df["Python"], df["Machine_Learning"])
plt.title("Python vs Machine Learning Marks")
plt.xlabel("Python Marks")
plt.ylabel("Machine Learning Marks")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# 4. Pie Chart
plt.figure()
performance_counts = df["Performance"].value_counts()
plt.pie(performance_counts, labels=performance_counts.index, autopct="%1.1f%%")
plt.title("Students by Performance Category")
plt.tight_layout()
plt.savefig("pie_chart.png")
plt.show()

# 5. Box Plot
plt.figure()
plt.boxplot([df["Math"], df["Python"], df["Machine_Learning"]], tick_labels=["Math", "Python", "Machine_Learning"])
plt.title("Spread of Marks in All Subjects")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("box_plot.png")
plt.show()
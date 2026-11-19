# Write
with open("notes.txt", "w") as f:
    f.write("Day 2 Task\n")
    f.write("Task is done\n")

# Read
with open("notes.txt", "r") as f:
    print(f.read())

# Append
with open("notes.txt", "a") as f:
    f.write("New line added\n")
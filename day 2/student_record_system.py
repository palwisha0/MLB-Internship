import json, os
FILE_NAME = "students.json"

def load_students():
    if not os.path.exists(FILE_NAME):
        return []
    try:
        with open(FILE_NAME, "r") as f:
            return json.load(f)
    except:
        return []

def save_students(students):
    with open(FILE_NAME, "w") as f:
        json.dump(students, f, indent=4)

students = load_students()
while True:
    print("\n1.Add 2.View 3.Search 4.Update 5.Delete 6.Exit")
    ch = input("Choice: ")
    try:
        if ch == "1":
            roll = int(input("Roll: "))
            name = input("Name: ")
            marks = float(input("Marks: "))
            students.append({"roll": roll, "name": name, "marks": marks})
            save_students(students)
            print("Added & Saved!")
        elif ch == "2":
            for s in students: print(s)
        elif ch == "3":
            roll = int(input("Search Roll: "))
            found = [s for s in students if s["roll"] == roll]
            print(found if found else "Not Found")
        elif ch == "4":
            roll = int(input("Update Roll: "))
            for s in students:
                if s["roll"] == roll:
                    s["name"] = input("New Name: ")
                    s["marks"] = float(input("New Marks: "))
                    save_students(students)
                    print("Updated!")
        elif ch == "5":
            roll = int(input("Delete Roll: "))
            students = [s for s in students if s["roll"] != roll]
            save_students(students)
            print("Deleted!")
        elif ch == "6":
            break
    except ValueError:
        print("Invalid Input! Please enter number.")
import json

# Python dict -> JSON file (dump)
student = {"name": "Palwasha", "age": 22, "marks": 80}
with open("data.json", "w") as f:
    json.dump(student, f, indent=4)

# JSON file -> Python dict (load)
with open("data.json", "r") as f:
    data = json.load(f)
    print(data)
    print(data["name"])
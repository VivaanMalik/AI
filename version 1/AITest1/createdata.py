import json
from random import randint

data = {"4 digit": [],
        "5 digit": [],
        "6 digit": [],
        "7 digit": [],}

for _ in range(1000):
    data["4 digit"].append(randint(1000, 9999))

for _ in range(1000):
    data["5 digit"].append(randint(10000, 99999))

for _ in range(1000):
    data["6 digit"].append(randint(100000, 999999))

for _ in range(1000):
    data["7 digit"].append(randint(1000000, 9999999))

with open('AITest1\\dataset2.json', 'w') as f:
    json.dump(data, f)
import random
import matplotlib.pyplot as plt

point_distances = {
    "x" : 9,
    "y" : 4,
    "z" : 1,
}

N = sum(point_distances.values())

def get_point_from_rand(result):
    threshold = 0
    for p in point_distances:
        threshold += point_distances[p]
        if result <= threshold:
            return p

count = {
    "x" : 0,
    "y" : 0,
    "z" : 0,
}

for i in range(1000):
    res = random.randint(1, N)
    point = get_point_from_rand(res)
    count[point] += 1

plt.bar([key for key in count], [count[key] for key in count], color='green')
plt.xticks([key for key in count], [key for key in count])
plt.show()

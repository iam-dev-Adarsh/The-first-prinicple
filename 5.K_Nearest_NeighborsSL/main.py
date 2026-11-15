import math
import random

K = 5  #(5 fold)
COLORS = ['Blue (class 0)', 'Red (class 1)']

def euclidean_distance(a,b):
    return math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2)

def generate_data():
    data = []
    clusters = [
        {'cx': 3, 'cy': 3, 'label': 0},
        {'cx': 7, 'cy': 7, 'label': 1},
    ]
    for c in clusters:
        for _ in range(25):
            data.append({
                'x': c['cx'] + (random.random()-0.5) * 4,
                'y': c['cy'] + (random.random()-0.5) * 4,
                'label': c['label'],

            })
    return data

def knn_predict(query,data,k):
    sorted_data = sorted(data, key=lambda p: euclidean_distance(query, p))
    neighbors = sorted_data[:k]
    votes = [0,0]
    for p in neighbors:
        votes[p['label']] += 1
    return votes.index(max(votes)), votes, neighbors 

data = generate_data()
query = {'x': 5, 'y': 4.5}

print(f"Training data: Class 0 has 25, Class 1 has 25")
print(f"Query point: ({query['x']}, {query['y']})\n")

predicted, votes, neighbors = knn_predict(query,data,K)

print(f"K= {K} nearest neighbors:")
sorted_data = sorted(data, key=lambda p: euclidean_distance(query,p))
for i, p in enumerate(sorted_data[:K]):
    d = euclidean_distance(query,p)
    print(f" {i + 1}th nearest: Class {p['label']}, distance {d:.2f}")

print(f"\nVote results: Class 0 {votes[0]} votes, Class 1 {votes[1]} votes")
print(f"Prediction: Class {predicted}")

print("Completed!!")
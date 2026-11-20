# K-Means - Pure Python implementation

import math 
import random

def euclidean_distance(p1,p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) **2)

def kmeans(data, k , max_iter = 100):
    centroids = random.sample(data, k )
    assignments = [0] * len(data)
    for _ in range(max_iter):
        for i , point in enumerate(data):
            min_dist = float('inf')
            for j , c in enumerate(centroids):
                dist = euclidean_distance(point , c)
                if dist < min_dist:
                    min_dist = dist
                    assignments[i] = j
        new_centroids = []
        for c in range(k):
            members = [data[i] for i in range(len(data)) if assignments[i] == c]
            if members:
                new_centroids.append([
                    sum(p[0] for p in members) / len(members),
                    sum(p[1] for p in members) / len(members)
                ])
            else:
                new_centroids.append(centroids[c])
        centroids = new_centroids
    return {'centroids': centroids, 'assignments': assignments}

data = [(1,1), (1.5,2), (8,8), (9,7)]
result = kmeans(data, 2, 5) 
print("Centroids: ", result['centroids'])         

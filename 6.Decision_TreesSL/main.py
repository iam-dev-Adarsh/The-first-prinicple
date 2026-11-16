import math
import random

MAX_DEPTH = 4
MIN_SAMPLES = 4

#--------------Generate moon-shaped data-----------------------------------------------
def generate_data():
    data = []
    for i in range(60):
        t = math.pi * i / 60
        data.append({'x': math.cos(t) * 3 + 5 + (random.random()-0.5)*1.2,
                        'y': math.sin(t) * 3 + 5 + (random.random()-0.5)*1.2, 'label': 0})

        data.append({'x': math.cos(t + math.pi) * 3 + 5 + (random.random()-0.5)*1.2,
                        'y': math.sin(t + math.pi) * 3 + 5 + (random.random()-0.5)*1.2,
                        'label': 1})
    return data

#-------Gini coefficeint-------------------------------------------------------------
def gini(labels):
    if not labels: return 0
    n = len(labels)
    counts = {} 
    for l in labels: counts[l] = counts.get(l,0) + 1
    return 1 - sum((c/n)**2 for c in counts.values())

#-------------Best split---------------------------------------------------------------
def best_split(data):
    base = gini([d['label'] for d in data])
    best_gain, best_feat, best_thresh = -1, None, None
    for feat in ['x', 'y']:
        vals = sorted(set(d[feat] for d in data))
        for i in range(len(vals) - 1):
            thresh = (vals[i] + vals[i+1]) / 2
            left = [d for d in data if d[feat] <= thresh]
            right = [d for d in data if d[feat] > thresh]
            if not left or not right: continue
            gain = base - (
                (len(left) / len(data)) * gini([d['label'] for d in left])
                + (len(right) / len(data)) * gini([d['label'] for d in right])
            )
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh
    return best_gain, best_feat, best_thresh

#----------Build Tree---------------------------------------------------------------------
def build_tree(data, depth = 0):
    labels = [d['label'] for d in data]
    majority = max(set(labels), key=labels.count)
    if depth >= MAX_DEPTH or len(data) < MIN_SAMPLES or len(set(labels)) == 1:
        return {'leaf': True, 'label': majority, 'count':len(data)}
    gain,feat,thresh = best_split(data)
    if feat is None:
        return {'leaf': True, 'label': majority, 'count': len(data)}
    left = [d for d in data if d[feat] <= thresh]
    right = [d for d in data if d[feat] > thresh]
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'gain': gain,
            'left':build_tree(left, depth + 1), 'right': build_tree(right, depth + 1)}

def predict(node,point):
    if node['leaf']: return node['label']
    return predict(node['left'] if point[node['feat']] <= node['thresh'] else node['right'],point)

def tree_depth(node):
    if node['leaf']: return 0
    return 1 + max(tree_depth(node['left']), tree_depth(node['right']))

def print_tree(node, indent = 0):
    if node['leaf']:
        print(' ' * indent + f"-> Predict: class {node['label']} ({node['count']} samples)")
        return
    print(' ' * indent + f'[{node['feat']} <= {node['thresh']:.2f}]  gain= {node['gain']:.4f}')
    print_tree(node['left'], indent + 4)
    print_tree(node['right'], indent + 4)


#------------------- Main program--------------------------------------------------
data = generate_data()
tree = build_tree(data)

print(f"Data: {len(data)} points, max depth {MAX_DEPTH}\n")
print("Decision tree structure:")
print_tree(tree)

acc = sum(predict(tree,d) == d['label'] for d in data) / len(data)
print(f"\nTree depth: {tree_depth(tree)}")
print(f"Training accuracy: {acc*100:.1f}%")

# predict a few test points
tests = [{'x': 5, 'y': 7}, {'x': 5, 'y': 3}, {'x': 8, 'y': 5}]
print("\nTest predictions:")
for p in tests:
    print(f" ({p['x']}, {p['y']}) -> class {predict(tree,p)}")


















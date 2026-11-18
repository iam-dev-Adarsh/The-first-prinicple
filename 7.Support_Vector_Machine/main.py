# SVM - Pure python implementation

import math 
import random

C = 1.0
LR = 0.005
EPOCHS = 5000    # previously 500 but the accuracy was 50.6% hence increased

#---------Generate linealry separable data-----------------------------
def generate_data():
    data = []
    for _ in range(30):
        x1 = 2 + random.random() * 3
        x2 = 2 + random.random() * 3

        data.append({'x': x1, 'y': x2, 'label': -1})
        data.append({'x': x1+3, 'y': x2+2+(random.random()-0.5)*1.5, 'label': 1})

    return data

#---------------------Svm sub-gradien training
def train_svm(data):
    w = [0.1, 0.1]
    b = 0.0
    for epoch in range(EPOCHS):
        dw = [0.0, 0.0]
        db = 0.0
        total_loss = 0
        for d in data:
            margin = d['label'] * (w[0]*d['x'] + w[1]*d['y'] + b)
            hinge = max(0, 1 - margin)
            total_loss += hinge
            if margin < 1:
                dw[0] -= C * d['label'] * d['x']
                dw[1] -= C * d['label'] * d['y']
                db -= C * d['label']
        w[0] -= LR * (w[0] + dw[0] / len(data))
        w[1] -= LR * (w[1] + dw[1] / len(data))
        
        b -= LR * (db / len(data)) 
        if epoch % 100 == 0 :
            norm = math.sqrt(w[0]**2 + w[1]**2)
            margin_w = 2 / norm if norm  > 1e-6 else 0
            print(f"Epoch {epoch}: Hinge loss = {total_loss/len(data):.4f}, Margin Width = {margin_w:.3f}")
    return w ,b

#-----Prediction-----------------------------------------------------------------------
def predict(w,b,point) :
    score = w[0] * point['x'] + w[1] * point['y'] + b 
    return 1 if score >= 0 else -1

#----------------Support vectors-----------------------------------------------
def find_support_vectors(w, b, data, eps=0.3):
    return [d for d in data if abs(d['label'] * (w[0]*d['x'] + w[1]*d['y'] + b) - 1) < eps]           

#---------------------Main program------------------------------------------------
data = generate_data()
print(f"Training data: {len(data)} points (linearly separable)\n")

w, b = train_svm(data)

norm = math.sqrt(w[0]**2 + w[1]**2)
margin = 2 / norm if norm > 1e-6 else 0
svs = find_support_vectors(w , b, data)

acc = sum(predict(w,b,d) == d['label'] for d in data) / len(data)
print(f"\n=========Training Results=========")
print(f"Weight vector: w = [{w[0]:.3f}, {w[1]:.3f}]")
print(f"Bias: b = {b:.3f}")
print(f"Margin width: {margin:.3f}")
print(f"Number of support vectors: {len(svs)}")
print(f"Training accuracy: {acc*100:.1f}%")

print("\nTest predictions: ")
tests = [{'x': 3, 'y': 3} , {'x': 7, 'y': 7}, {'x': 5, 'y': 5}]
for p in tests:
    score = w[0]*p['x'] + w[1]*p['y'] + b
    print(f" ({p['x']}, {p['y']}) ->  Decision value = {score:.2f} -> Predicted class= {predict(w,b,p)}")

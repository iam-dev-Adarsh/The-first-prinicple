# Softmax Regression - pure python implementation

import math
import random
import time
import matplotlib.pyplot as plt


LEARNING_RATE = 0.05
EPOCHS = 300
STEP_DELAY = 0.01  # seconds
CLASS_COUNT = 3

COLORS = ["#5470c6", "#ee6666", "#3ba272"]


def softmax(logits):
    largest = max(logits)
    exponentials = [math.exp(value - largest) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def generate_data():
    data = []

    for _ in range(30):
        data.append({
            "x1": 2 + (random.random() - 0.5) * 3,
            "x2": 2 + (random.random() - 0.5) * 3,
            "y": 0,
        })
        data.append({
            "x1": 8 + (random.random() - 0.5) * 3,
            "x2": 2 + (random.random() - 0.5) * 3,
            "y": 1,
        })
        data.append({
            "x1": 5 + (random.random() - 0.5) * 3,
            "x2": 8 + (random.random() - 0.5) * 3,
            "y": 2,
        })

    return data


def predict(weights, biases, x1, x2):
    logits = [
        weights[class_id][0] * x1
        + weights[class_id][1] * x2
        + biases[class_id]
        for class_id in range(CLASS_COUNT)
    ]
    probabilities = softmax(logits)
    return probabilities.index(max(probabilities))


def calculate_loss(data, weights, biases):
    total_loss = 0.0

    for point in data:
        logits = [
            weights[class_id][0] * point["x1"]
            + weights[class_id][1] * point["x2"]
            + biases[class_id]
            for class_id in range(CLASS_COUNT)
        ]
        probabilities = softmax(logits)
        total_loss -= math.log(probabilities[point["y"]] + 1e-9)

    return total_loss / len(data)


def calculate_gradients(data, weights, biases):
    weight_gradients = [[0.0, 0.0] for _ in range(CLASS_COUNT)]
    bias_gradients = [0.0] * CLASS_COUNT

    for point in data:
        logits = [
            weights[class_id][0] * point["x1"]
            + weights[class_id][1] * point["x2"]
            + biases[class_id]
            for class_id in range(CLASS_COUNT)
        ]
        probabilities = softmax(logits)

        for class_id in range(CLASS_COUNT):
            target = 1.0 if class_id == point["y"] else 0.0
            error = probabilities[class_id] - target

            weight_gradients[class_id][0] += error * point["x1"]
            weight_gradients[class_id][1] += error * point["x2"]
            bias_gradients[class_id] += error

    count = len(data)
    weight_gradients = [
        [gradient / count for gradient in row]
        for row in weight_gradients
    ]
    bias_gradients = [gradient / count for gradient in bias_gradients]

    return weight_gradients, bias_gradients


def draw(data, weights, biases, epoch, loss):
    background_points = [[] for _ in range(CLASS_COUNT)]

    coordinate = 0.0
    while coordinate <= 10:
        other_coordinate = 0.0
        while other_coordinate <= 10:
            class_id = predict(
                weights, biases, coordinate, other_coordinate
            )
            background_points[class_id].append(
                (coordinate, other_coordinate)
            )
            other_coordinate += 0.35
        coordinate += 0.35

    plt.clf()
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    if epoch == 0:
        plt.title("Random initial decision regions")
    else:
        plt.title(f"Epoch {epoch}  Loss={loss:.4f}")

    for class_id in range(CLASS_COUNT):
        region = background_points[class_id]
        plt.scatter(
            [point[0] for point in region],
            [point[1] for point in region],
            color=COLORS[class_id],
            alpha=0.15,
            s=6,
        )

        class_points = [
            point for point in data if point["y"] == class_id
        ]
        plt.scatter(
            [point["x1"] for point in class_points],
            [point["x2"] for point in class_points],
            color=COLORS[class_id],
            label=f"Class {class_id}",
        )

    plt.legend()
    plt.pause(0.001)


def train():
    data = generate_data()

    weights = [
        [
            (random.random() - 0.5) * 0.5,
            (random.random() - 0.5) * 0.5,
        ]
        for _ in range(CLASS_COUNT)
    ]
    biases = [
        (random.random() - 0.5) * 0.5
        for _ in range(CLASS_COUNT)
    ]

    print(f"Data: {len(data)} points")
    plt.ion()
    draw(data, weights, biases, 0, float("inf"))
    time.sleep(0.6)

    for epoch in range(1, EPOCHS + 1):
        loss = calculate_loss(data, weights, biases)
        weight_gradients, bias_gradients = calculate_gradients(
            data, weights, biases
        )

        for class_id in range(CLASS_COUNT):
            for feature_id in range(2):
                weights[class_id][feature_id] -= (
                    LEARNING_RATE
                    * weight_gradients[class_id][feature_id]
                )
            biases[class_id] -= (
                LEARNING_RATE * bias_gradients[class_id]
            )

        draw(data, weights, biases, epoch, loss)

        if epoch % 10 == 0:
            correct = sum(
                predict(weights, biases, point["x1"], point["x2"])
                == point["y"]
                for point in data
            )
            accuracy = correct / len(data)
            print(
                f"Epoch {epoch}: Loss={loss:.4f}, "
                f"Accuracy={accuracy * 100:.1f}%"
            )

        time.sleep(STEP_DELAY)

    correct = sum(
        predict(weights, biases, point["x1"], point["x2"]) == point["y"]
        for point in data
    )
    accuracy = correct / len(data)

    print(
        f"\nTraining complete! "
        f"Final accuracy: {accuracy * 100:.1f}%"
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    train()
import argparse
import os
import csv
import math
import numpy as np


def read_tsv(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((row["text"], int(row["label"])))
    print(f"Read {len(rows)} rows from {file_path}")
    return rows

def read_wordlists(pos_path, neg_path):
    with open(pos_path, "r", encoding="utf-8") as f:
        positive_words = f.read().strip().split("\n")
    with open(neg_path, "r", encoding="utf-8") as f:
        negative_words = f.read().strip().split("\n")
    return positive_words, negative_words

def read_data(train_path, dev_path, test_path):
    train_data = read_tsv(train_path)
    dev_data = read_tsv(dev_path)
    test_data = read_tsv(test_path)
    return train_data, dev_data, test_data

def encode_data(data, positive_words, negative_words):
    pairs = []
    positive_word_idx = {word: idx for idx, word in enumerate(positive_words)}
    negative_word_idx = {word: idx for idx, word in enumerate(negative_words)}

    for text, label in data:
        vector = np.zeros(403, dtype=float)

        individual_words = text.split()
        pos_count = 0
        neg_count = 0

        for word in individual_words:
            if word in positive_word_idx:
                index = positive_word_idx[word]
                vector[index] += 1
                pos_count += 1
            if word in negative_word_idx:
                index = 200 + negative_word_idx[word]
                vector[index] += 1
                neg_count += 1

        # Extra features
        vector[400] = len(individual_words)
        vector[401] = pos_count
        vector[402] = neg_count

        pairs.append((vector, label))
    return pairs

class LogisticRegression:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0.0

    def sigmoid(self, z):
        if z >= 0:
            exp_neg = np.exp(-z)
            return 1 / (1 + exp_neg)
        else:
            exp_pos = np.exp(z)
            return exp_pos / (1 + exp_pos)

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)

    def predict_label(self, x):
        return 1 if self.predict(x) >= 0.5 else 0

    def calculate_loss(self, x, y):
        eps = 1e-15
        y_hat = self.predict(x)
        y_hat = min(max(y_hat, eps), 1 - eps)
        return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

    def calculate_gradient(self, x, y):
        y_hat = self.predict(x)
        err = y_hat - y
        dw = err * x
        db = err
        return dw, db

def accuracy(y_pred, y_true):
    acc = sum(int(p == t) for p, t in zip(y_pred, y_true))
    return acc/len(y_true)

def train_model(model, train_pairs, dev_pairs, num_steps, learning_rate):
    n = len(train_pairs)
    for step in range(num_steps):
        features, label = train_pairs[step % n]
        dw, db = model.calculate_gradient(features, label)
        model.weights -= learning_rate * dw
        model.bias -= learning_rate * db


def run_experiments(train_pairs, dev_pairs):
    learning_rates = [0.1, 0.01, 0.001]
    steps_list = [10000, 100000, 500000]
    results = []

    for lr in learning_rates:
        for steps in steps_list:
            num_features = len(train_pairs[0][0])
            model = LogisticRegression(num_features=num_features)
            train_model(model, train_pairs, dev_pairs, steps, lr)
            y_pred = [model.predict_label(x) for x, _ in dev_pairs]
            y_true = [y for _, y in dev_pairs]
            dev_acc = accuracy(y_pred, y_true)
            results.append((lr, steps, dev_acc))
    return results


def analyze_weights(model, positive_words, negative_words):
    word_weights = []

    for idx, word in enumerate(positive_words):
        weight = model.weights[idx]
        word_weights.append((word, weight))

    for idx, word in enumerate(negative_words):
        weight = model.weights[200 + idx]
        word_weights.append((word, weight))

    word_weights.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 50 words by weight magnitude:")
    for word, weight in word_weights[:50]:
        print(f"{word}: {weight:.4f}")


def test_model(model, test_pairs):
    y_pred = [model.predict_label(x) for x, _ in test_pairs]
    y_true = [y for _, y in test_pairs]
    test_acc = accuracy(y_pred, y_true)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Process Yelp review datasets")
    parser.add_argument("--train_path", default="data" + os.sep + "train.tsv")
    parser.add_argument("--dev_path", default="data" + os.sep + "dev.tsv")
    parser.add_argument("--test_path", default="data" + os.sep + "test.tsv")
    parser.add_argument("--positive_words", default="data" + os.sep + "positive.txt")
    parser.add_argument("--negative_words", default="data" + os.sep + "negative.txt")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=500000)
    parser.add_argument("--run_experiments", action="store_true")

    args = parser.parse_args()
    # Read all data files
    train_data, dev_data, test_data = read_data(args.train_path, args.dev_path, args.test_path)

    # Read wordlists
    positive_words, negative_words = read_wordlists(args.positive_words, args.negative_words)

    # Encode the data
    train_pairs = encode_data(train_data, positive_words, negative_words)
    dev_pairs = encode_data(dev_data, positive_words, negative_words)
    test_pairs = encode_data(test_data, positive_words, negative_words)

    if args.run_experiments:
        run_experiments(train_pairs, dev_pairs)
    else:
        num_features = len(train_pairs[0][0])
        model = LogisticRegression(num_features=num_features)
        train_model(model, train_pairs, dev_pairs, args.num_steps, args.learning_rate)
        analyze_weights(model, positive_words, negative_words)
        test_model(model, test_pairs)


if __name__ == "__main__":
    main()

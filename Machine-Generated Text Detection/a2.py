import argparse
import csv
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from treeviz import visualize_tree

def extract_bigrams(tokens):
    """
    Takes list of tokens like ["a", "b", "c"] and produces a list of 2-tuples like [("a", "b"), ("b", "c")]
    """
    bigrams = []                            #defined a list named bigrams that will store the grouped tokens
    for i in range(len(tokens) - 1):        #for each number from 0 to len(tokens)-1
        bigram = (tokens[i], tokens[i + 1]) #storing group of two tokens as tuples
        bigrams.append(bigram)              #appending this "bigram" value to "bigrams list"
    return bigrams                          #return a list of tuples, where each tuple contains group of two tokens. [("a","b"),("b","c"),("c","d")]

def load_tsv(file_path):
    """
    Load data from TSV file and extract bigrams.
    """
    data = []                                         # data would be stored in this list and form of data would be (bigrams,label)
    with open(file_path, 'r', encoding='utf-8') as f: #opens file and reads according to utf-8 encoding
        reader = csv.DictReader(f, delimiter='\t')    #Dictreader reads data from .tsx files as {column_name_1:val,column_name_2:val}. For our example it is {'label':'1','tokens':'whole_string'}
        for row in reader:                            # now for each row in reader
            label = int(row['label'])                 #convert the given label into integer
            tokens = row['tokens'].split() if row['tokens'] else [] #convert the string into list of words/tokens. The condition says if row['tokens'] exists then split it into list of words/token ['crfr','fwrfr'] etc else keep it []
            bigrams = extract_bigrams(tokens)         #using antoher function "extract_bigrams" to group tokens into group of two
            data.append((bigrams, label))             # Appended the bigrams:[('fr','fer'),('fer','sdf'),('sdf','acx')] list with label, so finally data becomes[([('fr','fer'),('fer','sdf'),('sdf','acx')],1)]
    return data

def build_vocabulary(train_data, vocabulary_size, min_count):
    """
    Builds a shared vocabulary of bigrams from the training data.

    Args:
        train_data: List of (bigrams, label) pairs
        vocabulary_size: Maximum vocabulary size (top-k most frequent bigrams)
        min_count: Minimum frequency for vocabulary inclusion

    Returns:
        dict: Vocabulary mapping bigrams to indices
    """
    # Count bigrams from training data
    bigram_counts = Counter()

    for bigrams, label in train_data:
        bigram_counts.update(bigrams)

    # Filter by min_count and take top vocabulary_size bigrams
    filtered_bigrams = {bigram: count for bigram, count in bigram_counts.items() if count >= min_count}
    # Sort by count (descending), then by bigram tuple (ascending) for deterministic ordering
    top_bigrams = sorted(filtered_bigrams.items(), key=lambda x: (-x[1], x[0]))[:vocabulary_size]

    # Create vocabulary mapping
    vocab = {bigram: idx for idx, (bigram, count) in enumerate(top_bigrams)}
    print(f"Built vocabulary: {len(bigram_counts)} total bigrams, {len(filtered_bigrams)} meet min_count={min_count}, using top {len(vocab)}")
    return vocab


def load_and_process_data(vocabulary_size, min_count):
    """
    Load pre-processed data from TSV files, build vocabulary, and extract bigrams.

    Args:
        vocabulary_size: Maximum vocabulary size (top-k most frequent bigrams)
        min_count: Minimum frequency for vocabulary inclusion

    Returns:
        tuple: (train_data, dev_data, test_data, vocab) where each data is list of (bigrams, label) pairs
    """
    print("Loading pre-processed data...")

    # Load data splits (bigrams extracted automatically)
    train_data = load_tsv('data/train.tsv') #Train data is loaded in form of [([('a','b'),('b','c'),('c','d')],1)]
    dev_data = load_tsv('data/dev.tsv')     #Dev_data data is loaded in form of [([('a','b'),('b','c'),('c','d')],1)]
    test_data = load_tsv('data/test.tsv')   #Test_data is loaded in form of [([('a','b'),('b','c'),('c','d')],1)]

    print(f"Loaded {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test examples")

    # Build vocabulary from training data
    print("Building vocabulary...")
    vocab = build_vocabulary(train_data, vocabulary_size, min_count) #vocab is built using build_vocabulary function, which of form {('a','b'):0,('b','c'):1}, where key is bigrams and value is index(0th index is most occuring)
    print(f"Vocabulary size: {len(vocab)}")

    return train_data, dev_data, test_data, vocab #returning all values of train_data, dev_data, test_data, vocab


def baseline_classifier_accuracy(dev_data):
    """
    Evaluate majority class baseline on development data.

    Args:
        dev_data: List of (bigrams, label) pairs

    Returns:
        float: Accuracy of majority class baseline
    """
    # TODO: Implement majority class baseline
    labels=[]
    for _,label in dev_data:
        labels.append(label)
    
    count_of_0=labels.count(0)
    count_of_1=labels.count(1)
    majority=count_of_0 if count_of_0>count_of_1 else count_of_1

    accuracy=majority/len(dev_data)
    print(f"Baseline (majority class) accuracy: {accuracy:.4f}")
    return accuracy


def vectorize_data(data, vocab):
    # Labels: pack them all together into a vector
    y = np.array([label for _, label in data])        # A vector y array, containing labels of the data
    # TODO: put bigram counts into X
    X = np.zeros((len(data), len(vocab)))             # A feature vector X, initalized to 0, with row=len(data) and column=len(vocab)
    for i,(list_of_tuples,label) in enumerate(data):  #enumerating through data to get index i and list_of_tuples present in data
        for bigram in list_of_tuples:                 #now for each bigram in that list_of_tuples
            if bigram in vocab:                       #Check if it exists in vocab
                j=vocab[bigram]                       # if yes then get the value from vocab and assign it to j, where key is bigram
                X[i,j]+=1                             #increment feature vector at row value i and column value j by 1
    return X, y


def analyze_lr_weights(model, vocab, top_k=30):
    """
    Analyze and print top features from logistic regression weights.

    Args:
        model: Trained LogisticRegression model
        vocab: 2-gram vocabulary
        top_k: Number of top features to show
    """
    print(f"\n=== Top {top_k} Logistic Regression Features ===")
    # TODO: Print top positive and negative weight features
    weights=model.coef_[0]

    inv_vocab = {idx: bigram for bigram, idx in vocab.items()} #inverted the vocab to have idx:bigram
    top_pos_indices = np.argsort(weights)[-top_k:][::-1] #Sorting the weights using argsort, then [-top_k:] get last 30 elements from sorted array, then [::-1] makes these 30 elements into descending order
    top_neg_indices = np.argsort(weights)[:top_k]        #Getting top 30 elements from sorted array

    print("\n--- Top Positive Weights (indicate label=1, e.g. human-generated) ---")
    for idx in top_pos_indices:
        print(f"{inv_vocab[idx]}:{weights[idx]:.4f}")

    print("\n--- Top Negative Weights (indicate label=0, e.g. machine-written) ---")
    for idx in top_neg_indices:
        print(f"{inv_vocab[idx]}:{weights[idx]:.4f}")


def analyze_dt_features(model, vocab, top_k=30):
    """
    Analyze and print most important features from decision tree.

    Args:
        model: Trained DecisionTreeClassifier model
        vocab: 2-gram vocabulary
        top_k: Number of top features to show
    """
    print(f"\n=== Top {top_k} Decision Tree Features ===")
    # TODO: Print top important features
    importances = model.feature_importances_
    top_indices = importances.argsort()[::-1][:top_k]

    for idx in top_indices:
        feature = vocab[idx] if isinstance(vocab, list) else list(vocab.keys())[idx]
        importance = importances[idx]
        print(f"{feature}: {importance:.4f}")

def test_final_model(model, X_test, y_test):
    test_acc = model.score(X_test, y_test)
    print(f"Final test accuracy: {test_acc:.4f}")
    return test_acc

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Machine-generated text detection with multiple models")

    # Data arguments
    parser.add_argument("--vocabulary_size", type=int, default=10000,
                        help="Maximum vocabulary size (top-k most frequent bigrams)")
    parser.add_argument("--min_count", type=int, default=5,
                        help="Minimum frequency for vocabulary inclusion")

    # Model selection
    parser.add_argument("--model", choices=["lr", "dt"], default="lr",
                        help="Model type: lr (Logistic Regression) or dt (Decision Tree)")

    # Logistic Regression hyperparameters
    parser.add_argument("--lr_C", type=float, default=1.0,
                        help="Regularization strength for Logistic Regression")
    parser.add_argument("--lr_penalty", default="None",
                        help="Penalty type for Logistic Regression (None, l1, l2)")
    parser.add_argument("--lr_max_iter", type=int, default=100,
                        help="Maximum iterations for Logistic Regression")

    # Decision Tree hyperparameters
    parser.add_argument("--dt_max_depth", type=str, default="5",
                        help="Maximum depth for Decision Tree")
    parser.add_argument("--dt_min_samples_split", type=int, default=50,
                        help="Minimum samples to split for Decision Tree")
    parser.add_argument("--dt_criterion", default="gini", choices=["gini", "entropy"],
                        help="Criterion for Decision Tree")

    # Analysis options
    parser.add_argument("--top_k", type=int, default=30,
                        help="Number of top features to analyze")

    args = parser.parse_args()
    try:
        args.dt_max_depth = int(args.dt_max_depth)
    except ValueError:
        args.dt_max_depth = None

    # Log arguments (filter model-specific ones)
    print("=== Configuration ===")
    print(f"Bigram vocabulary size: {args.vocabulary_size}")
    print(f"Bigram minimum count: {args.min_count}")
    print(f"top_k for analysis: {args.top_k}")
    print(f"Model: {args.model}")

    if args.model == "lr":
        print(f"LR regularization type: {args.lr_penalty}")
        print(f"LR regularization weight (lower => more regularization): {args.lr_C}")
        print(f"LR max iterations: {args.lr_max_iter}")
    elif args.model == "dt":
        print(f"DT max depth: {args.dt_max_depth}")
        print(f"DT min samples split: {args.dt_min_samples_split}")
        print(f"DT criterion: {args.dt_criterion}")

    print("=" * 20)

    # Load and process data (with caching)
    train_data, dev_data, test_data, vocab = load_and_process_data(
        vocabulary_size=args.vocabulary_size,
        min_count=args.min_count
    )

    # Data analysis
    baseline_classifier_accuracy(dev_data)

    # Vectorize data (bigrams already extracted and cached)
    print("\nVectorizing data...")
    X_train, y_train = vectorize_data(train_data, vocab)
    X_dev, y_dev = vectorize_data(dev_data, vocab)
    X_test, y_test = vectorize_data(test_data, vocab)

    # Train model based on selection
    print(f"\nTraining {args.model.upper()} model...")

    if args.model == "lr":
        # Logistic Regression
        model = LogisticRegression(
            solver="lbfgs" if args.lr_penalty == "None" else "liblinear",
            C=args.lr_C,
            penalty=None if args.lr_penalty == "None" else args.lr_penalty,
            max_iter = args.lr_max_iter,
            random_state=42
        )
        model = model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        dev_acc = model.score(X_dev, y_dev)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Development accuracy: {dev_acc:.4f}")

        # Analyze weights
        analyze_lr_weights(model, vocab, top_k=args.top_k)

    elif args.model == "dt":
        # Decision Tree
        model = DecisionTreeClassifier(
            max_depth=args.dt_max_depth,
            min_samples_split=args.dt_min_samples_split,
            criterion=args.dt_criterion,
            random_state=42
        )
        model = model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        dev_acc = model.score(X_dev, y_dev)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Development accuracy: {dev_acc:.4f}")

        # Analyze feature importance
        analyze_dt_features(model, vocab, top_k=args.top_k)

        # Visualize tree structure (for interpretation)
        visualize_tree(model, vocab, args.dt_max_depth)

    # Uncomment for final test evaluation (Part 5)
    test_final_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
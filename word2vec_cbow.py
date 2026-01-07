import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import random
from collections import Counter



class Vocabulary:
    """
    Handles building the vocabulary and word-to-index mappings.
    (This class is complete and requires no changes.)
    """
    def __init__(self, filepath, min_freq=5):
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        self.word_counts = Counter()
        # self.sampling_probs = None # No longer needed
        self._build(filepath, min_freq)

    def _build(self, filepath, min_freq):
        print("Building vocabulary...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                self.word_counts.update(words)

        # Create word2idx and idx2word
        idx = 1  # Start from 1, as 0 is for <UNK>
        for word, count in self.word_counts.items():
            if count >= min_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

        # We still store word counts, but not for sampling
        vocab_words = set(self.word2idx.keys())
        self.word_counts = Counter(
            {word: count for word, count in self.word_counts.items()
             if word in vocab_words}
        )
        # Add <UNK> count
        unk_count = sum(
            count for word, count in self.word_counts.items()
            if word not in vocab_words
        )
        self.word_counts['<UNK>'] = unk_count
        print(f"Vocabulary size: {len(self.word2idx)}")

    def get_idx(self, word):
        """Returns the index of a word, or the <UNK> index."""
        return self.word2idx.get(word, self.word2idx['<UNK>'])

    def __len__(self):
        return len(self.word2idx)


class CBOWDataset(Dataset):
    """
    Custom PyTorch Dataset for CBOW.
    This dataset reads a text file and generates (context, target) pairs.
    """

    def __init__(self, filepath, vocab, window_size):
        self.vocab = vocab
        self.window_size = window_size
        self.data = []
        print(f"Loading and processing data from {filepath}...")

        # For CBOW, the goal is to predict a center word (the "target")
        # from the surrounding words (the "context").
        #
        # For a window size of 2, in the sentence "The quick brown fox jumps",
        # when "brown" is the target, the context is ["The", "quick", "fox", "jumps"], and
        # we would add (["The", "quick", "fox", "jumps"], "brown") to self.data after we've
        # converted each word (a string) into its numeric vocabulary index.
        # For initial testing purposes, you can uncomment this line to create one dummy sample
        # self.data.append(([1, 2, 4, 5], 3))

        # TODO:
        # 1. Open the `filepath` and read each line.
        # 2. For each line (sentence):
        #    a. Split the line into `words`.
        #    b. Convert `words` to `token_indices` `self.vocab`.
        #    c. Find every valid instance within the sentence and append it to
        
        size_of_context = window_size * 2
        index_of_unk = self.vocab.get_idx('<UNK>')

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                
                if len(words) < 1:
                    continue

                token_indices = [self.vocab.get_idx(w) for w in words]
                for i in range(len(token_indices)):
                    index_of_target = token_indices[i]

                    start = max(0, i - self.window_size)
                    end = min(len(token_indices), i + self.window_size + 1)

                    context_indices = []
                    for j in range(start, end):
                        if j == i:
                            continue
                        context_indices.append(token_indices[j])

                    if len(context_indices) < size_of_context:
                        context_indices += [index_of_unk] * (size_of_context - len(context_indices))

                    self.data.append((context_indices, index_of_target))

        if len(self.data) == 0:
            print("WARNING: `self.data` is empty.")
        else:
            print(f"Created {len(self.data)} (context, target) pairs.")

    def __len__(self):
        """Returns the total number of (context, target) pairs."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one training sample.
        A sample consists of:
        - context (list of word indices)
        - target (a single word index)
        """
        # TODO
        # 1. Get the `context_indices` and `target_index` from `self.data[idx]`.
        # 2. Return a 2-tuple with:
        #    - torch.tensor(context_indices, dtype=torch.long)
        #    - torch.tensor(target_index, dtype=torch.long)
        context_indices, target_index = self.data[idx]
        return (
            torch.tensor(context_indices, dtype=torch.long),
            torch.tensor(target_index, dtype=torch.long)
        )


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOWModel, self).__init__()

        # Embedding module for context words
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)

        # The "output" is a linear layer that projects the averaged
        # context embedding to the vocabulary size.
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        # Initialize input embedding weights
        nn.init.xavier_uniform_(self.input_embedding.weight)

    def forward(self, context):
        """
        Forward pass.
        Args:
            context (Tensor): (batch_size, window_size * 2) tensor of context word indices.
        Returns:
            logits (Tensor): (batch_size, vocab_size) scores for each word in vocab.
        """

        # TODO
        # 1. Get embedding for `context` using `self.input_embedding`.
        # 2. Average the context embedding over the window dimension
        #    (Hint: `torch.mean()`).
        # 3. Pass the `context_vector` through the `self.output_layer`.
        # 4. Return `logits`, which should have the shape [batch_size, vocab_size].
        #    To produce a prediction, we would apply the softmax to this to get a probability
        #    distribution over vocaulary items on the second axis, but for producing loss during
        #    training, all we need is the logits.
        embeddings = self.input_embedding(context)        
        context_vector = torch.mean(embeddings, dim=1)    
        logits = self.output_layer(context_vector)    
        return logits


def save_embeddings(model, vocab, filepath):
    """
    Saves the input word embeddings in the word2vec .txt format.
    """
    print(f"Saving embeddings to {filepath}...")
    # Get the input embeddings from the model
    embeddings = model.input_embedding.weight.data.cpu().numpy()
    vocab_size = len(vocab)
    embed_dim = embeddings.shape[1]

    with open(filepath, 'w', encoding='utf-8') as f:
        # Write the header (vocab_size, embed_dim)
        f.write(f"{vocab_size} {embed_dim}\n")

        # Write each word and its vector
        for i in range(vocab_size):
            word = vocab.idx2word[i]
            vector = ' '.join(f"{x:.6f}" for x in embeddings[i])
            f.write(f"{word} {vector}\n")
    print("Embeddings saved.")


def find_similar_words(model, vocab, word, k=5):
    """
    Finds the k most similar words to a given word using cosine similarity
    on the input embeddings.
    """
    model.eval()  # Set model to evaluation mode

    word_idx = vocab.get_idx(word)
    if word_idx == vocab.get_idx('<UNK>'):
        print(f"Word '{word}' not in vocabulary.")
        return

    # Get the embedding for the target word
    word_vec = model.input_embedding.weight[word_idx].unsqueeze(0)  # (1, embed_dim)

    # Get all embeddings
    all_embeds = model.input_embedding.weight  # (vocab_size, embed_dim)

    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(word_vec, all_embeds)  # (vocab_size,)

    # Get the top k+1 indices (k+1 because the word itself will be #1)
    top_k_vals, top_k_indices = torch.topk(cos_sim, k + 1)

    print(f"--- Top {k} similar words to '{word}' ---")
    for i in range(1, k + 1):  # Start from 1 to skip the word itself
        idx = top_k_indices[i].item()
        similar_word = vocab.idx2word[idx]
        similarity = top_k_vals[i].item()
        print(f"{i}. {similar_word} (Similarity: {similarity:.4f})")
    print("---------------------------------")


def main():
    """
    Main function to orchestrate the data loading, training, and evaluation.
    """
    # For reproducibility, set random seed
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # --- Hyperparameters ---
    EMBED_DIM = 50
    WINDOW_SIZE = 2
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 0.003
    MIN_FREQ = 5
    print("Beginning run. Hyperparameters:")
    print("-" * 50)
    print(f"| EMBED_DIM: {EMBED_DIM}")
    print(f"| WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"| BATCH_SIZE: {BATCH_SIZE}")
    print(f"| EPOCHS: {EPOCHS}")
    print(f"| LEARNING_RATE: {LEARNING_RATE}")
    print(f"| MIN_FREQ: {MIN_FREQ}")
    print("-" * 50)

    # Use CUDA or Apple Silicon if available, fall back to CPU by default
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Build Vocabulary
    data_file = 'data/brown.txt'
    vocab = Vocabulary(data_file, min_freq=MIN_FREQ)

    # 3. Create Dataset and DataLoader
    train_dataset = CBOWDataset(
        filepath=data_file,
        vocab=vocab,
        window_size=WINDOW_SIZE
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # 4. Initialize Model, Loss, and Optimizer
    vocab_size = len(vocab)
    model = CBOWModel(vocab_size, EMBED_DIM).to(device)

    # Use CrossEntropyLoss for multiclass classification.
    # Note that this loss function expects LOGITS as its first argument, even though it wants LABELS as its
    # second argument. This is done for reasons of numerical stability.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        # The loop variables are updated for CBOW
        for i, (context, target) in enumerate(train_loader):
            # Move data to the selected device
            context = context.to(device)
            target = target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: get logits
            logits = model(context)

            # Calculate loss
            # criterion(logits, target)
            # - logits shape: (batch_size, vocab_size)
            # - target shape: (batch_size,)
            loss = criterion(logits, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 200 == 0:  # Print progress every 200 batches
                print(f'Epoch {epoch + 1}/{EPOCHS}, '
                      f'Batch {i + 1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'--- End of Epoch {epoch + 1}, Average Loss: {avg_loss:.4f} ---')

        # Find similar words at the end of each epoch
        # These dummy words are based on the dummy data
        find_similar_words(model, vocab, 'person')
        find_similar_words(model, vocab, 'good')

    # 6. Post-Training
    print("Training complete.")

    # Save the learned embeddings
    save_embeddings(model, vocab, 'cbow_embeddings.txt')

    # Find some similar words
    find_similar_words(model, vocab, 'person')
    find_similar_words(model, vocab, 'good')

    extra_words=["hard","glutton","female","aesthetic","book"]
    for each_word in extra_words:
        find_similar_words(model, vocab, each_word)
if __name__ == "__main__":
    main()

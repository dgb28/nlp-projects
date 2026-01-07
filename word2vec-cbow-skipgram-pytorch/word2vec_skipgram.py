import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import random
from collections import Counter


class Vocabulary:
    def __init__(self, filepath, min_freq=5):
        self.word2idx = {'<UNK>': 0} #Initialized a word to index map
        self.idx2word = {0: '<UNK>'} #Initialized a index to word map
        self.word_counts = Counter() #Created a counter that count words
        self.sampling_probs = None   #sampling_probs is None
        self._build(filepath, min_freq) #builds vocabulary

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

        # Store filtered counts for negative sampling
        vocab_words = set(self.word2idx.keys())
        self.word_counts = Counter({word: count for word, count in self.word_counts.items()
                                    if word in vocab_words})
        # Add <UNK> count
        unk_count = sum(count for word, count in self.word_counts.items()
                        if word not in vocab_words)
        self.word_counts['<UNK>'] = unk_count

        print(f"Vocabulary size: {len(self.word2idx)}")

    def create_sampling_table(self, power=0.75):
        """
        Creates the probability distribution for negative sampling.
        Uses the unigram distribution raised to the 0.75 power, as in the paper.
        """
        print("Creating negative sampling table...")
        # Get word frequencies in the order of idx2word
        counts = torch.zeros(len(self.idx2word))
        for i in range(len(self.idx2word)):
            word = self.idx2word[i]
            counts[i] = self.word_counts[word]

        # Raise to the 0.75 power
        powered_counts = torch.pow(counts, power)

        # Normalize to get probabilities
        self.sampling_probs = powered_counts / torch.sum(powered_counts)

    def get_idx(self, word):
        """Returns the index of a word, or the <UNK> index."""
        return self.word2idx.get(word, self.word2idx['<UNK>'])

    def __len__(self):
        return len(self.word2idx)


class SkipGramDataset(Dataset):
    """
    Custom PyTorch Dataset for Skip-gram.
    This dataset reads a text file and generates (center, context) pairs.
    """

    def __init__(self, filepath, vocab, window_size, neg_samples):
        self.vocab = vocab
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.sampling_probs = vocab.sampling_probs
        self.data = []

        # Process the data during dataset initialization so that self.data holds (center, context) pairs where
        # each item in the pair is a word index from the vocabulary. We will generate negative samples on demand.
        #
        # For example, for a window size of 2, in the sentence "The dog likes to bark", when "likes" i the center word,
        # we would add ("likes", "The"), ("likes", "dog"), ("likes", "to"), ("likes", "bark") added to self.data
        print(f"Loading and processing data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if len(words) < 2:
                    continue

                # Convert words to indices
                token_indices = [self.vocab.get_idx(w) for w in words]

                # Create (center, context) pairs
                for i in range(len(token_indices)):
                    center_word_idx = token_indices[i]

                    # Take care not to overstep the boundaries of the sentence
                    start = max(0, i - self.window_size)
                    end = min(len(token_indices), i + self.window_size + 1)

                    for j in range(start, end):
                        if i == j:  # Skip the center word itself
                            continue
                        context_word_idx = token_indices[j]
                        self.data.append((center_word_idx, context_word_idx))

        print(f"Created {len(self.data)} (center, context) pairs.")

    def __len__(self):
        """Returns the total number of (center, context) pairs."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one training sample.
        A sample consists of:
        - center_word (index)
        - pos_context_word (index)
        - neg_context_words (k indexes, dynamically sampled)
        """
        # Get the positive (center, context) pair
        center, pos_context = self.data[idx]

        # Sample k negative context words
        # `torch.multinomial` samples indices from the distribution
        neg_context = torch.multinomial(
            self.sampling_probs,
            num_samples=self.neg_samples,
            replacement=True
        )

        # We return Tensors, and the DataLoader will batch them
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(pos_context, dtype=torch.long),
            neg_context.long()
        )


class SkipGramNegativeSampling(nn.Module):
    """
    The Skip-gram model with Negative Sampling.
    """

    def __init__(self, vocab_size, embed_dim):
        super(SkipGramNegativeSampling, self).__init__()

        # Two embedding layers are used: one for center words, one for context words
        self.center_embedding = nn.Embedding(vocab_size, embed_dim)
        self.context_embedding = nn.Embedding(vocab_size, embed_dim)

        # Initialize weights (e.g., Xavier initialization)
        # This can help with training stability
        nn.init.xavier_uniform_(self.center_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def forward(self, center, pos_context, neg_context):
        """
        Forward pass.
        Args:
            center (Tensor): (batch_size,) tensor of center word indices.
            pos_context (Tensor): (batch_size,) tensor of positive context word indices.
            neg_context (Tensor): (batch_size, k) tensor of negative context word indices.
        """
        # 1. Get embeddings
        # (batch_size, embed_dim)
        center_embeds = self.center_embedding(center)
        # (batch_size, embed_dim)
        pos_context_embeds = self.context_embedding(pos_context)
        # (batch_size, k, embed_dim)
        neg_context_embeds = self.context_embedding(neg_context)

        # 2. Calculate scores

        # Positive score:
        # We want the dot product of center_embeds and pos_context_embeds
        # (batch_size, 1, embed_dim) * (batch_size, embed_dim, 1) -> (batch_size, 1, 1)
        # A simpler way is element-wise multiply and sum
        # (batch_size, embed_dim) * (batch_size, embed_dim) -> (batch_size, embed_dim)
        # sum(dim=1) -> (batch_size,)
        pos_score = torch.sum(center_embeds * pos_context_embeds, dim=1)

        # Negative score:
        # We want the dot product of center_embeds with *each* of the k neg_context_embeds
        # center_embeds:            (batch_size, embed_dim)
        # We need to reshape it to  (batch_size, 1, embed_dim) to use broadcasting
        center_embeds_expanded = center_embeds.unsqueeze(1)

        # neg_context_embeds:      (batch_size, k, embed_dim)
        # Broadcasting rules:
        #    (batch_size, 1, embed_dim)
        #  * (batch_size, k, embed_dim)
        # -> (batch_size, k, embed_dim)
        # This does an element-wise multiplication, "copying" the center vector k times.

        # We then sum over the last dimension (embed_dim) to get the dot products
        # (batch_size, k, embed_dim) -> sum(dim=2) -> (batch_size, k)
        neg_score = torch.sum(center_embeds_expanded * neg_context_embeds, dim=2)

        # We return the raw scores (logits), not sigmoids, because nn.BCEWithLogitsLoss
        # automatically applies the sigmoid for us for reasons of numeric stability.
        return pos_score, neg_score


def save_embeddings(model, vocab, filepath):
    """
    Saves the center word embeddings in the classic word2vec .txt format.
    """
    print(f"Saving embeddings to {filepath}...")
    # Get the center embeddings from the model
    embeddings = model.center_embedding.weight.data.cpu().numpy()
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
    Finds the k most similar words to a given word using cosine similarity.
    """
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode

    word_idx = vocab.get_idx(word)
    if word_idx == vocab.get_idx('<UNK>'):
        print(f"Word '{word}' not in vocabulary.")
        return

    # Get the embedding for the target word
    word_vec = model.center_embedding.weight[word_idx].unsqueeze(0)  # (1, embed_dim)

    # Get all embeddings
    all_embeds = model.center_embedding.weight  # (vocab_size, embed_dim)

    # Calculate cosine similarity
    # F.cosine_similarity computes (v1 . v2) / (||v1|| * ||v2||)
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
    # Feel free to adjust these
    EMBED_DIM = 50 # Dimensionality of word embeddings
    WINDOW_SIZE = 2  # Context window size (words to the left/right)
    BATCH_SIZE = 128 # Number of (center, context) pairs per batch
    EPOCHS = 10  # Number of training epochs
    LEARNING_RATE = 0.003  # Optimizer learning rate
    MIN_FREQ = 5  # Minimum word frequency to be included in vocab
    NEG_SAMPLES = 5  # Number of negative samples (k) for each positive pair
    print("Beginning run. Hyperparameters:")
    print("-" * 50)
    print(f"| EMBED_DIM: {EMBED_DIM}")
    print(f"| WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"| BATCH_SIZE: {BATCH_SIZE}")
    print(f"| EPOCHS: {EPOCHS}")
    print(f"| LEARNING_RATE: {LEARNING_RATE}")
    print(f"| MIN_FREQ: {MIN_FREQ}")
    print(f"| NEG_SAMPLES: {NEG_SAMPLES}")
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
    vocab = Vocabulary('data/brown.txt', min_freq=MIN_FREQ)
    vocab.create_sampling_table()

    # 3. Create Dataset and DataLoader
    train_dataset = SkipGramDataset(
        filepath='data/brown.txt',
        vocab=vocab,
        window_size=WINDOW_SIZE,
        neg_samples=NEG_SAMPLES
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # 4. Initialize Model, Loss, and Optimizer
    vocab_size = len(vocab)
    model = SkipGramNegativeSampling(vocab_size, EMBED_DIM).to(device)

    # BCEWithLogitsLoss is perfect for this. It combines Sigmoid + BinaryCrossEntropy
    # and is numerically stable (handles log(0) issues).
    criterion = nn.BCEWithLogitsLoss()

    # Adam is a good default optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Set model to training mode. Internally, this sets `self.training` to True. Modules may check this
        # to change their behavior. For example, Dropout modules are disabled when self.training is False.
        # This is not necessary for our particular model, but it is good practice to always use this and
        # its counterpart, model.eval(), which sets self.training to False.
        model.train()

        # We'll maintain an average loss per epoch just for logging purposes
        total_loss = 0.0

        for i, (center, pos_context, neg_context) in enumerate(train_loader):
            # Move data to the selected device
            center = center.to(device)
            pos_context = pos_context.to(device)
            neg_context = neg_context.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            pos_score, neg_score = model(center, pos_context, neg_context)

            # Calculate loss
            # We want pos_score to be high (target=1) and neg_score to be low (target=0)

            # Create labels: 1s for positive, 0s for negative
            pos_labels = torch.ones_like(pos_score, device=device)
            neg_labels = torch.zeros_like(neg_score, device=device)

            # Calculate loss for positive and negative samples separately
            pos_loss = criterion(pos_score, pos_labels)
            neg_loss = criterion(neg_score, neg_labels)

            loss = pos_loss + neg_loss

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

        find_similar_words(model, vocab, 'person')
        find_similar_words(model, vocab, 'good')

    # 6. Post-Training
    print("Training complete.")

    # Save the learned embeddings
    save_embeddings(model, vocab, 'skipgram_embeddings.txt')

    # Find some similar words (using the dummy data vocab)
    find_similar_words(model, vocab, 'person')
    find_similar_words(model, vocab, 'good')


if __name__ == "__main__":
    main()

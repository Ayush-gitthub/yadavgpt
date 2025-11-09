🚀 Building a GPT-style Language Model From Scratch
This guide will walk you through the entire process of building a small, GPT-like language model from the ground up using Python and PyTorch. We will start with the simplest possible model and incrementally add the key components of the Transformer architecture—the engine behind models like ChatGPT.
By the end, you will have a working, deep-learning model that you've built and trained yourself, and you'll understand the core mechanics of how Large Language Models (LLMs) work.
📋 Table of Contents
Prerequisites: Setting Up Your Environment
Step 1: The Foundation - Data Preparation
Step 2: The Simplest Model - A Bigram Baseline
Step 3: Teaching the Model - The Training Loop
Step 4: The Leap - The Self-Attention Mechanism
Step 5: Scaling Up - The Full Transformer Block
Step 6: The Final Model - Building a Deep Network
The Complete Code & How to Run It
🛠️ Prerequisites: Setting Up Your Environment
Before we start, let's get your workshop ready.
1. Python (Version 3.11 Recommended)
The deep learning ecosystem has many dependencies. For the best compatibility with the latest PyTorch and CUDA versions, Python 3.11 is the most stable and recommended choice.
Install Python 3.11: Download it from the official Python website. Crucially, check the box that says "Add python.exe to PATH" during installation.
2. Project Setup
Create a project folder and set up a virtual environment. This keeps your project dependencies isolated.
code
Bash
# In your terminal
mkdir yadavgpt
cd yadavgpt

# Create a virtual environment using Python 3.11
py -3.11 -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate
3. PyTorch with GPU Support
To train our model efficiently, we need a GPU. An NVIDIA RTX 30-series or 40-series card is ideal. We will install PyTorch with CUDA support.
code
Bash
# Install the GPU version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Note: If this command fails, it's almost certainly because of a Python version mismatch. Following the steps above to use Python 3.11 is the most reliable fix.
📜 Step 1: The Foundation - Data Preparation
Every machine learning model begins with data. Our goal is to train a model that can write like Shakespeare.
Goal: Turn the complete works of Shakespeare into a stream of numbers that our model can learn from.
Download the Data: We'll use the "TinyShakespeare" dataset. Create a file named input.txt in your project folder and save the contents from this link: tinyshakespeare/input.txt.
Tokenization: Our model can't read characters like 'A' or 'z'. We need to convert them to numbers. We will use a simple character-level tokenizer.
We find every unique character in the text (our vocabulary).
We create a mapping from each character to a unique integer (e.g., 'a' -> 32).
Train/Validation Split: We'll use 90% of the data to train the model and hold back the final 10% to validate its performance on data it has never seen.
🤖 Step 2: The Simplest Model - A Bigram Baseline
We start with the most basic language model imaginable to create a baseline.
Concept: A bigram model predicts the next character by only looking at the single character immediately preceding it. It's essentially learning the probability of which character follows which (e.g., 'q' is almost always followed by 'u').
In neural network terms, this is just a single embedding table where each token looks up the probabilities for the next token.
The initial, untrained model will generate complete gibberish, but this proves our plumbing works.
Initial Output (Untrained):
code
Code
c-p$Af3T zAb'i'XJe!Qy,H:vOqj!E?xP?z,Lg,vepFlCQk...
🧠 Step 3: Teaching the Model - The Training Loop
A model is useless until it learns. The training loop is the process where the model improves.
Core Components:
Loss Function: We use Cross-Entropy Loss to measure how "wrong" the model's predictions are compared to the actual next character. A high loss means the model was very surprised; a low loss means it was confident and correct.
Optimizer: We use the AdamW optimizer. Its job is to look at the loss and slightly adjust the model's internal parameters (its "weights") to make the loss smaller.
The Loop: We repeatedly:
Grab a random batch of data.
Ask the model for a prediction.
Calculate the loss.
Ask the optimizer to update the model.
After thousands of steps, the model's loss decreases, and its generated text starts to look less random. It learns to form word-like structures.
Output (Trained Bigram Model, Loss ~2.5):
code
Code
CETh a he he s, ppprinke, ad w he, he, so mind w wey hat te, nof my ouck...
```It's still nonsense, but it has learned about spaces and basic character pairings!
✨ Step 4: The Leap - The Self-Attention Mechanism
Our bigram model's biggest flaw is its lack of memory. It has no context. The revolutionary idea behind the Transformer is self-attention.
Intuition: To understand a word in a sentence, we pay attention to the other words that give it meaning. Self-attention allows our model to do the same. For each token, it looks at all the previous tokens and decides which ones are most important for understanding its own meaning.
How it works (The Head):
For each token, we create three vectors: a Query ("what am I looking for?"), a Key ("what information do I contain?"), and a Value ("what I will provide if you pay attention to me").
The model calculates an "affinity score" between each token's Query and every other token's Key.
These scores are turned into weights.
The final output for a token is a weighted sum of all the Value vectors. It's now enriched with context from the other tokens it paid attention to.
🏗️ Step 5: Scaling Up - The Full Transformer Block
A single attention head is good, but a real Transformer is more powerful. We upgrade to a full Block.
Key Upgrades:
Multi-Head Attention: We run several attention heads in parallel. Each head can learn to focus on different linguistic patterns, and we combine their results.
Feed-Forward Network: After attention gathers information, a simple neural network "thinks" about the information gathered for each token individually.
Residual Connections: We add "shortcuts" or "skip connections" that allow the input of a layer to be added to its output. This is crucial for training deep networks by allowing the learning signal to flow more easily.
Layer Normalization: We add normalization layers to keep the training process stable as data flows through the deep network.
These components—attention and feed-forward, with residual connections and normalization—form one Transformer Block.
🏛️ Step 6: The Final Model - Building a Deep Network
The final step is to stack these Transformer Blocks on top of each other to create a deep network. The output of one block becomes the input to the next.
A shallow network might learn words.
A deep network can learn grammar, sentence structure, and even thematic coherence.
We define a final set of hyperparameters to control the size of our model:
n_embd = 384: The size of our token embeddings.
n_head = 6: The number of parallel attention heads.
n_layer = 6: The number of Transformer Blocks to stack.
This results in a model with ~10.79 million parameters. It is still tiny compared to GPT-3, but it is a genuine, deep Transformer model.
Final Output (Trained Transformer, Loss ~1.5):
code
Code
PRINCE:
To her in this, and tell her, my lord,
To give him with my hands that he was slain upon him,
And see his son, which I cannot help the gods
My life of his son: I am a Roman, that we shall
Be so found in our father's death.
The difference is staggering. The model now generates coherent, grammatically plausible text that is thematically consistent with the dataset.
💻 The Complete Code & How to Run It
Here is the complete, final model.py script.
code
Python
# model.py (Complete Final Version)

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters for the new, bigger model ---
batch_size = 64 # How many independent sequences will we process in parallel?
block_size = 256 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # A lower learning rate is often better for larger models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # 384-dimensional embeddings
n_head = 6 # 6 attention heads
n_layer = 6 # 6 transformer blocks
dropout = 0.2
# ----------------------------------------------------

print(f"Using device: {device}")
torch.manual_seed(1337)

# --- Data Loading ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# ---------------------

# --- Data Loading Function ---
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # Move data to the correct device
    return x, y
# ---------------------

# --- Helper function to estimate loss ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out
# -----------------------------------------

# --- Neural Network Modules ---

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- The Full, Deep Language Model ---
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Model and Optimizer Initialization ---
model = LanguageModel()
m = model.to(device)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
# --------------------------------------

# --- The Training Loop ---
print("\n--- Starting Training ---")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# -------------------------

# --- Generate from the TRAINED model ---
print("\n--- Text Generated by Final Model ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=1000)[0].tolist())
print(generated_text)
# --------------------------------------
To Run This Code:
Follow the setup steps in the Prerequisites section.
Save the code above as model.py.
Make sure input.txt is in the same directory.
Run the script from your activated virtual environment:
code
Bash
python model.py
Congratulations! You have now walked the entire path from a blank file to a working Transformer, and hopefully, you've gained a deep, practical understanding of how these incredible models are built.

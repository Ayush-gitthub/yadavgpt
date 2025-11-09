# generate.py (for inference only)

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters (MUST BE THE SAME AS TRAINING) ---
# We don't need all the training-specific ones, but the model architecture ones are crucial.
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ----------------------------------------------------

# We need to load the tokenizer mappings
# In a real application, you would save these to a file as well.
# For now, we will re-create them.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# ----------------------------------------------------


# --- Re-create the Model Architecture ---
# You need to have the Python classes for the model available.
# The saved file 'model.pth' ONLY contains the weights, not the code for the model's structure.

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
        B,T,C = x.shape; k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1); wei = self.dropout(wei)
        v = self.value(x); out = wei @ v; return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd); self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)); return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size); self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x)); return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd); self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape; tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb; x = self.blocks(x); x = self.ln_f(x); logits = self.lm_head(x)
        if targets is None: loss = None
        else:
            B, T, C = logits.shape; logits = logits.view(B*T, C); targets = targets.view(B*T)
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
# ----------------------------------------------------


# --- Load the Model ---
print("Loading model...")
model = LanguageModel()
m = model.to(device)
# Load the state dictionary
m.load_state_dict(torch.load('model.pth'))
# Set the model to evaluation mode
m.eval()
print("Model loaded successfully.")
# ---------------------

# --- Generate Text ---
print("\n--- Generating new text ---")
# context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with a single newline character
prompt = "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?\n"
encoded_prompt = encode(prompt)

context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
generated_text = decode(m.generate(context, max_new_tokens=1000)[0].tolist())
print(generated_text)
# ---------------------
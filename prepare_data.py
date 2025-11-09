import torch # We'll use PyTorch for our model

# Open and read the text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("--- First 200 characters of the dataset ---")
print(text[:200])
print("\n--- Dataset Statistics ---")
print(f"Length of dataset in characters: {len(text):,}")

# Get all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"\nAll unique characters: {''.join(chars)}")
print(f"Vocabulary size: {vocab_size}")

# Create a mapping from characters to integers (and vice-versa)
stoi = { ch:i for i,ch in enumerate(chars) } # string-to-integer
itos = { i:ch for i,ch in enumerate(chars) } # integer-to-string

# Encoder: takes a string, outputs a list of integers
encode = lambda s: [stoi[c] for c in s]

# Decoder: takes a list of integers, outputs a string
decode = lambda l: ''.join([itos[i] for i in l])

# Let's test it
test_string = "hello world"
encoded_string = encode(test_string)
decoded_string = decode(encoded_string)

print("\n--- Tokenization Example ---")
print(f"Original string: '{test_string}'")
print(f"Encoded representation: {encoded_string}")
print(f"Decoded back to string: '{decoded_string}'")

# Encode the entire text dataset and store it in a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print("\n--- Full Dataset Encoded ---")
print(f"Encoded data tensor shape: {data.shape}")
print(f"First 100 encoded tokens: {data[:100]}")

# Split up the data into train and validation sets
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print("\n--- Data Splitting ---")
print(f"Training data length: {len(train_data):,}")
print(f"Validation data length: {len(val_data):,}")
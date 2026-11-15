# Listing 2.5 A dataset for batched inputs and targets
# Tokenizer
from importlib.metadata import version
import tiktoken
import torch 
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) #Tokenize the entire text


#Uses sliding window in chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

# Returns the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

# Return a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# Listing 2.6 A data loader to generate batches with input-with pairs

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, 
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  # initializes the tokenizer 
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # creates dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, # drop_last=True drop the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
        num_workers=num_workers # the number of CPU processes to use for preprocessing
    )

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

"""dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False)
data_iter = iter(dataloader)  # convert dataloader into a Python iterator to fetch the next entry via Python's built-in next() function
inputs, targets = next(data_iter)
print("Input:\n", inputs)
print("\nTargets:\n", targets)"""

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)
data_iter = iter(dataloader)  # convert dataloader into a Python iterator to fetch the next entry via Python's built-in next() function
inputs, targets = next(data_iter)
print("Token Ids:\n", inputs)
print("\nInputs shape :\n", inputs.shape)
#the token ID tensor is 8 × 4 dimensional, meaning that the data batch consists of eight text samples with four tokens each.


# embedding layer to embed these token IDs into 256-dimensional vectors
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
# The 8 × 4 × 256–dimensional tensor output shows that each token ID is now embedded as a 256-dimensional vector.


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
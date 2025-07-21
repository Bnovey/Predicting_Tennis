import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import random
import math

class FixedPrefixDataset(Dataset):
    def __init__(self, match_tensors, labels, min_prefix=10, prefixes_per_match=20, max_prefix=False):
        self.match_tensors = match_tensors
        self.labels = labels
        self.min_prefix = min_prefix
        self.prefixes_per_match = prefixes_per_match
        self.num_matches = len(match_tensors)
        self.max_prefix = max_prefix
        
        if self.max_prefix:
            # Store all possible prefixes for every match, from min_prefix to full length
            self.samples = []
            for match_idx, seq in enumerate(match_tensors):
                L = seq.size(0)
                for t in range(self.min_prefix, L + 1):
                    prefix = seq[:t]
                    label = labels[match_idx]
                    self.samples.append((prefix, label))

    def __len__(self):
        if self.max_prefix:
            return len(self.samples)
        return self.num_matches * self.prefixes_per_match

    def __getitem__(self, idx):
        if self.max_prefix:
            return self.samples[idx]
        match_idx = idx // self.prefixes_per_match
        seq = self.match_tensors[match_idx]
        L = seq.size(0)
        t = random.randint(self.min_prefix, L)
        prefix = seq[:t]
        label = self.labels[match_idx]
        return prefix, label
    
# class AllPrefixesDataset(Dataset):
#     def __init__(self, match_tensors, labels, min_prefix=5):
#         """
#         match_tensors: list of [L_i, feat_dim] tensors
#         labels:        list or 1D tensor of 0/1
#         min_prefix:    smallest prefix length to use
#         """
#         self.samples = []
#         for match_idx, seq in enumerate(match_tensors):
#             L = seq.size(0)
#             for t in range(min_prefix, L + 1):
#                 prefix = seq[:t]
#                 label = labels[match_idx]
#                 self.samples.append((prefix, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

# Collate function remains the same:
def collate_fn(batch):
    sequences, labels = zip(*batch)         # list of [Li×F] tensors
    lengths = [s.size(0) for s in sequences]
    
    # pad up to the longest Li → [B×Lmax×F]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # build a [B×Lmax] mask: True where padding
    max_len  = padded.size(1)
    idx      = torch.arange(max_len).unsqueeze(0)       # [1×Lmax]
    lens     = torch.tensor(lengths).unsqueeze(1)       # [B×1]
    pad_mask = idx >= lens       
                           # [B×Lmax]
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, labels, pad_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class MatchOutcomeTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, padding_mask):
        x = self.input_proj(x)       # → [B, L, D]
        x = self.pos_enc(x)          # → [B, L, D]
        x = x.transpose(0, 1)

       
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )                             # → [L, B, D]

        # 4) back to [B, L, D]
        x = x.transpose(0, 1)

        # 5) masked mean‐pool over time
        #    - valid mask is 1 for real points, 0 for pads
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)  # → [B, L, 1]
        summed = (x * valid).sum(dim=1)                   # → [B, D]
        counts = valid.sum(dim=1).clamp(min=1)             # → [B, 1]
        pooled = summed / counts                          # → [B, D]

        # 6) classifier → logit per match
        #    (if you still have a Sigmoid in your classifier, this returns a probability;
        #     otherwise drop Sigmoid and use BCEWithLogitsLoss)
        out = self.classifier(pooled).squeeze(-1)         # → [B]
        return out
            



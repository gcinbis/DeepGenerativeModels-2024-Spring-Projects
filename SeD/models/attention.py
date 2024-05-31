#All You Need ... is Attention

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dimensionality=32):
        super(SelfAttention, self).__init__()
        self.heads = num_heads
        self.dim_head = dimensionality
        self.scale = dimensionality ** -0.5
        self.dim = dim

        # Define linear projection layers for queries, keys, and values
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # Define output projection layer
        self.final_output = nn.Linear(dim, dim)

    def forward(self, x):
        # Assume input x has shape [batch_size, seq_len, dim]

        # Linearly project queries, keys, and values
        q,k,v = self.to_qkv(x).chunk(3, dim=-1) # -1 for the last dimension

        # Reshape to [batch_size * heads, seq_len, dim_head]
        q = q.transpose(1, 2).reshape(-1, x.size(1), self.dim_head)
        k = k.transpose(1, 2).reshape(-1, x.size(1), self.dim_head)
        v = v.transpose(1, 2).reshape(-1, x.size(1), self.dim_head)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and combine heads
        attn_output = attn_output.reshape(-1, self.heads, x.size(1), self.dim_head)
        attn_output = attn_output.transpose(1, 2).reshape(-1, x.size(1), self.dim)

        # Apply output projection
        output = self.final_output(attn_output)

        # Add residual connection
        output += x

        return output
    

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        context_dim = context_dim or query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        #initialize qkv projection layers
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(context_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(context_dim, heads * dim_head, bias=False)
        
        #final output layer
        self.final_output = nn.Linear(heads * dim_head, query_dim)

    def forward(self, query, context):
        # Reshape and transpose query, key, and value tensors
        q = self.to_q(query).view(query.size(0), query.size(1), self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(context.size(0), context.size(1), self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(context.size(0), context.size(1), self.heads, self.dim_head).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(query.size(0), query.size(1), self.heads * self.dim_head)

        # Apply output projection
        output = self.final_output(attn_output)
        return output

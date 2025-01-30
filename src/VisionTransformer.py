import torch
from torch import nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim= 256):
        super().__init__()
        self.flatten_layer = nn.Flatten(start_dim=2, end_dim= 3)
        self.linear = nn.Linear(in_features= input_dim, out_features = output_dim)
    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.linear(x)
        return x
    

class PositionalEncodingAbs(nn.Module): #Absolute
    def __init__(self, num_patches= 16):
        super().__init__()
        n = num_patches ** (1/2) # 16 -> n=4
        sub_n = n ** (1/2) # 4 -> 2
        self.positional_encodings = torch.zeros(num_patches, 3, requires_grad= False).to(device)
        for i in range(num_patches):
            row = (i // n)
            column = (i % n)
            grid = (((row // sub_n)) * sub_n) + ((column // sub_n)) # The subgrid that i belongs to
            # print(i + 1, row + 1, column + 1, grid + 1)
            self.positional_encodings[i] = torch.tensor([row, column, grid]).to(device) / num_patches
        self.positional_encodings = torch.stack([self.positional_encodings] * 10, 0)
    def forward(self, x):
        x = torch.cat((x, self.positional_encodings), 2)
        # print(self.positional_encodings.shape)
        # print(x.shape)
        return x
    
class PositionalEncodingFreq(nn.Module): # Frequency-based
    def __init__(self, d_model= 256, max_seq_length= 16):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model).to(device)
        position = torch.arange(0, max_seq_length, dtype=torch.float).to(device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)

        pe[:, 0::2] = torch.sin(position * div_term).to(device)
        pe[:, 1::2] = torch.cos(position * div_term).to(device)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class ClassEmbedding(nn.Module):
    def __init__(self, embedding_dim= 256, batch_size= 10):
        super().__init__()
        self.class_embedding = torch.randn(1, embedding_dim).to(device)
        self.class_embedding = nn.Parameter(torch.stack([self.class_embedding] * batch_size, 0)).to(device)
        # print(self.class_embedding.shape)

    def forward(self, x):
        output = torch.cat([self.class_embedding, x], 1)
        return output
    
class PatchEmbeddingLayer(nn.Module):
    def __init__(self, grid_size, embedding_dim, input_dim, output_dim, freq_encoding= True):
        super().__init__()
        if freq_encoding:
            self.linear_projection = LinearProjection(input_dim= input_dim, output_dim= output_dim)
            self.positional_encoding = PositionalEncodingFreq()
        else:
            self.linear_projection = LinearProjection(input_dim= input_dim, output_dim= output_dim-3)
            self.positional_encoding = PositionalEncodingAbs()
        self.class_embedding = ClassEmbedding(embedding_dim= embedding_dim, batch_size= 10)


    def forward(self, x):
        x = self.linear_projection(x)
        x = self.positional_encoding(x)
        x = self.class_embedding(x)

        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim= 256, num_heads= 8, attn_dropout= 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_heads
        self.attn_dropout = attn_dropout

        self.layernorm = nn.LayerNorm(normalized_shape = embedding_dim)

        self.multiheadattention =  nn.MultiheadAttention(num_heads = num_heads,
            embed_dim = embedding_dim,
            dropout = attn_dropout,
            batch_first = True,
        )

    def forward(self, x):
        x = self.layernorm(x)
        output,_ = self.multiheadattention(query=x, key=x, value=x, need_weights=False)
        return output


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_size, mlp_dropout= 0):

        super().__init__()

        self.layernorm = nn.LayerNorm(normalized_shape= embedding_dim)

        self.linear = nn.Sequential(
            nn.Linear(in_features= embedding_dim, out_features= mlp_size),
            nn.GELU(),
            nn.Dropout(p= mlp_dropout),
            nn.Linear(in_features= mlp_size, out_features= embedding_dim),
            nn.Dropout(p= mlp_dropout)
        )

    def forward(self, x):

        x = self.layernorm(x)
        x = self.linear(x)

        return x
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim= 256, mlp_dropout= 0.1, attn_dropout= 0, mlp_size= 3072, num_heads= 8):
        super().__init__()

        self.attention = MultiHeadAttention(embedding_dim= embedding_dim, num_heads= num_heads, attn_dropout= attn_dropout)
        self.feedforward = MLP(embedding_dim = embedding_dim, mlp_size = mlp_size, mlp_dropout = mlp_dropout)

    def forward(self, x):
        x = self.attention(x) + x
        x = self.feedforward(x) + x

        return x
    
class ViT(nn.Module):
    def __init__(self, shape= [10, 16, 28, 28], freq_encoding= True, embedding_dim= 256, num_transformer_layers= 3, mlp_dropout = 0.1, attn_dropout = 0.0, mlp_size = 256, num_heads = 4, num_classes = 2):

        super().__init__()

        grid_size = shape[1]
        height, width = shape[2], shape[3]

        self.patch_embedding = PatchEmbeddingLayer(freq_encoding= freq_encoding, grid_size= grid_size, input_dim = height * width, output_dim= embedding_dim, embedding_dim= embedding_dim)
        self.transformer = nn.Sequential(
            *[Transformer(embedding_dim= embedding_dim, mlp_dropout= mlp_dropout, attn_dropout= attn_dropout, mlp_size= mlp_size, num_heads= num_heads) for _ in range(num_transformer_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape= embedding_dim),
            nn.Linear(in_features= embedding_dim, out_features= num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.classifier(x[:, 0])
        return x
    

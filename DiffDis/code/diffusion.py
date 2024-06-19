import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(num_heads, embed_dim, in_proj_bias=False)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output = self.attention(x)
        x = self.layernorm1(x + attn_output)
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x


class UNet_Transformer(nn.Module):
    def __init__(self, num_blocks=6, embed_dim=1280, num_heads=64, ff_dim=2048, text_query_dim=768):
        super(UNet_Transformer, self).__init__()
        self.text_query_projector = nn.Linear(text_query_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)]
        )
        self.linear_predictor = nn.Linear(embed_dim, text_query_dim)

    def forward(self, image_features, text_query):
        # Flatten image features (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height * Width)
        batch_size, channels, height, width = image_features.shape
        image_features = image_features.view(batch_size, channels, height * width)  # (Batch_Size, 320, Height * Width)

        # Project text query to the same dimension as image features
        text_query = self.text_query_projector(text_query)  # (Batch_Size, 768) -> (Batch_Size, 320)

        # Concatenate flattened image features and text query
        text_query = text_query.unsqueeze(2)  # (Batch_Size, 320) -> (Batch_Size, 320, 1)
        x = torch.cat((text_query, image_features), dim=2)  # (Batch_Size, 320, Height * Width + 1)
        x = x.transpose(1, 2)  # (Batch_Size, Height * Width + 1, 320)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Separate text query
        text_query = x[:, 0]  # (Batch_Size, 320)

        # Apply the linear predictor to the text query token
        predicted_text_query = self.linear_predictor(text_query)  # (Batch_Size, 768)

        return predicted_text_query


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768, is_upsample=False):
        super().__init__()
        channels = n_head * n_embd

        if is_upsample:
            self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
            self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        if not is_upsample:
            self.text_linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
            self.text_linear_geglu_2 = nn.Linear(4 * channels, channels)
            self.linearLayer = nn.Linear(d_context, channels)
            self.output_linearLayer = nn.Linear(channels, d_context)


    def prepare_input(self, aug_emb, hidden_text_query, latent):
        # Add the augmented embeddings and the hidden text query
        x = aug_emb + hidden_text_query

        # Apply the linear layer
        x = self.linearLayer(x)

        # Get the shape of latent
        batch_size, channels, height, width = latent.shape

        # Flatten the latent tensor
        latent = latent.view(batch_size, channels, height * width)

        # Reshape x to match the dimensions (Batch Size, Channels, 1)
        x = x.view(batch_size, -1, 1)

        # Concatenate x and latent along the spatial dimension
        x = torch.cat((x, latent), dim=2)

        return x

    def forward(self, x, context, aug_emb=None, hidden_text_query=None, is_upsample=True):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        # Get the shape of the input
        n, c, h, w = x.shape

        if aug_emb is not None and hidden_text_query is not None and not is_upsample:
            # Prepare the input for the attention block
            x = self.prepare_input(aug_emb, hidden_text_query, x)

        # Get the long skip connection
        if is_upsample:
            residue_long = x
        else:
            # Ensure x is split correctly into 1 and the remaining along the spatial dimension
            spatial_dim = x.shape[2] - 1  # The last dimension of x minus 1
            text_residue_long, residue_long = x.split([1, spatial_dim], dim=2)
            residue_long = residue_long.view(n, c, h, w)

            # (Batch_Size, Channels, Width * Height + 1) -> (Batch_Size, Channels, 1, Width * Height + 1)
            x = x.view(x.shape[0], x.shape[1], 1, -1)

        # Apply group normalization
        x = self.groupnorm(x)

        if not is_upsample:
            # (Batch_Size, Channels, 1, Width * Height + 1) -> (Batch_Size, Channels, Width * Height + 1)
            x = x.view(x.shape[0], x.shape[1], -1)
        else:
            # Apply the input convolution if the block is an upsample block
            # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
            x = self.conv_input(x)
            
            # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
            x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features) if upsample block
        # (Batch_Size, Features, Height * Width + 1) -> (Batch_Size, Height * Width + 1, Features) if not upsample block
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Check if the block is an upsample block, if not, split the output into two parts
        if not is_upsample:
            # (Batch_Size, Height * Width + 1, Features) -> two tensors of shape (Batch_Size, 1, Features) and (Batch_Size, Height * Width, Features)
            spatial_dim = x.shape[1] - 1  # The last dimension of x minus 1
            hidden_text_query, x = x.split([1, spatial_dim], dim=1)

        # Normalization + Cross-Attention with skip connection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        
        residue_short = x
        if not is_upsample:
            text_residue_short = hidden_text_query

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        if not is_upsample:
            # (Batch_Size, 1, Features) -> (Batch_Size, 1, Features)
            hidden_text_query = self.layernorm_3(hidden_text_query)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        if not is_upsample:
            # (Batch_Size, 1, Features) -> two tensors of shape (Batch_Size, 1, Features * 4)
            hidden_text_query, gate_text_query = self.text_linear_geglu_1(hidden_text_query).chunk(2, dim=-1)
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        if not is_upsample:
            # Element-wise product: (Batch_Size, 1, Features * 4) * (Batch_Size, 1, Features * 4) -> (Batch_Size, 1, Features * 4)
            hidden_text_query = hidden_text_query * F.gelu(gate_text_query)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features) if upsample block
        x = self.linear_geglu_2(x)
        if not is_upsample:
            # (Batch_Size, 1, Features * 4) -> (Batch_Size, 1, Features) if not upsample block
            hidden_text_query = self.text_linear_geglu_2(hidden_text_query)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        if not is_upsample:
            hidden_text_query += text_residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)            
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        if not is_upsample:
            # (Batch_Size, 1, Features) -> (Batch_Size, Features, 1)
            hidden_text_query = hidden_text_query.transpose(-1, -2)

            # Add the long skip connection to the output of the block
            hidden_text_query += text_residue_long
            
            # (Batch_Size, Features, 1) -> (Batch_Size, Features)
            hidden_text_query = hidden_text_query.squeeze(-1)          

            # Apply the output linear layer to the hidden text query
            hidden_text_query = self.output_linearLayer(hidden_text_query)

            return (x + residue_long, hidden_text_query)

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time, aug_emb=None, hidden_text_query=None, is_upsample=True):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                if aug_emb is not None and hidden_text_query is not None and not is_upsample:
                    x, hidden_text_query = layer(x, context, aug_emb, hidden_text_query, is_upsample)
                else:
                    x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        if not is_upsample:
            return x, hidden_text_query
                    
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, is_upsample=True)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, is_upsample=True)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160, is_upsample=True), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80, is_upsample=True)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80, is_upsample=True)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80, is_upsample=True), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40, is_upsample=True)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, is_upsample=True)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, is_upsample=True)),
        ])

        self.unet_transformer = UNet_Transformer()

    def forward(self, x, context, time, aug_emb):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        hidden_text_query = torch.zeros_like(aug_emb)
        for layers in self.encoders:
            # Get the output of the attention block
            x, hidden_text_query = layers(x, context, time, aug_emb, hidden_text_query, is_upsample=False)
            skip_connections.append(x)

        # Get the output of the middle block of the UNET
        x, hidden_text_query = self.bottleneck(x, context, time, aug_emb, hidden_text_query, is_upsample=False)

        # Pass through the UNet_Transformer
        hidden_text_query = self.unet_transformer(x, hidden_text_query)

        # Get the output of the upsampling blocks
        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x, hidden_text_query


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_time_embedding = TimeEmbedding(320)
        self.text_time_embedding = TimeEmbedding(192)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, image_time_embeddings, text_time_embeddings, text_query):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # image_time_embeddings: (1, 320)
        # text_time_embeddings: (1, 192)
        # text_query: (1, 768)

        # (1, 320) -> (1, 1280)
        image_time_embeddings = self.image_time_embedding(image_time_embeddings)

        # (1, 192) -> (1, 768)
        text_time_embeddings = self.text_time_embedding(text_time_embeddings)

        # Augment the text query with the text time embeddings
        aug_emb = text_query + text_time_embeddings
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        image_output, text_output = self.unet(latent, context, image_time_embeddings, aug_emb)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        image_output = self.final(image_output)
        
        # (Batch, 4, Height / 8, Width / 8)
        return image_output, text_output
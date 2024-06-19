import torch
import torch.nn as nn
from attention import CrossAttention
from attention import SelfAttention

class SemanticAwareFusionBlock(nn.Module):
    def __init__(self, channel_size_changer_input_nc=512):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, 1024) 

        self.channel_size_changer1 = nn.Conv2d(in_channels=channel_size_changer_input_nc, out_channels=128, kernel_size=1)
        self.reduce_channels2 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1)

        self.layer_norm_1 = nn.LayerNorm(128)
        self.layer_norm_2 = nn.LayerNorm(128)
        self.layer_norm_3 = nn.LayerNorm(128)

        self.self_attention = SelfAttention(128, num_heads=1, dimensionality=128)
        self.cross_attention = CrossAttention(128, heads=1, dim_head=128)

        self.GeLU = nn.GELU()

        #define 1x1 convolutions
        self.increase_channels1 = nn.Conv2d(256, 1024, 1)

    def forward(self, semantic_feature_maps, fs):
        # fs ( or sh for generated) have shape batch, 3 x 16 x 16
        #semantic feature maps  have shape batch x 1024 x 16 x 16
        final_permute_height = semantic_feature_maps.shape[2]
        final_permute_width = semantic_feature_maps.shape[3]
        
        #first handle S_h
        semantic_feature_maps = self.group_norm(semantic_feature_maps)

        #reduce the channel dimensions for the feature maps from 1024 to 128 for computation
        semantic_feature_maps = self.reduce_channels2(semantic_feature_maps)


        # Permute dimensions to rearrange the tensor
        semantic_feature_maps = semantic_feature_maps.permute(0, 2, 3, 1).contiguous().view(semantic_feature_maps.size(0), -1, semantic_feature_maps.size(1))

        #apply layer normalization
        semantic_feature_maps = self.layer_norm_1(semantic_feature_maps)

        #apply self attention
        semantic_feature_maps = self.self_attention(semantic_feature_maps) #returned has shape 1,196,128 for now
        #apply layer normalization
        query = self.layer_norm_2(semantic_feature_maps)

        #now handle fs or  sh
        #reduce the channel dimensions for the sh

        #make number of channels = 128 to be compatible with the semantic feature maps
        fs = self.channel_size_changer1(fs)

        #to use fs as residual, obtain a clone, 
        #note that gradient still accumulates in the original fs, so no problem
        fs_residual = fs.clone()

        #permute the dimensions
        fs = fs.permute(0, 2, 3, 1).contiguous().view(fs.size(0), -1, fs.size(1))

        #apply cross attention, query is the semantic feature maps and fs is the key and value
        out = self.cross_attention(query, fs)

        #apply layer normalization
        out = self.layer_norm_3(out)

        #apply GeLU
        out = self.GeLU(out)

        #permute the dimensions
        out = out.permute(0,2,1).contiguous().view(out.size(0), -1, final_permute_height, final_permute_width)

        #add the residual
        output = torch.cat((out,fs_residual), dim=1)

        #increase the channels back to 1024
        output = self.increase_channels1(output)
    
        return output


if __name__ == "__main__":
    model = SemanticAwareFusionBlock(channel_size_changer_input_nc=128)
    semantic = torch.randn(1, 1024, 16, 16)
    sh = torch.randn(1, 128, 32, 32)

    with torch.no_grad():
        out = model(semantic, sh)

import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Mlp(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(in_features=int(embed_dim * mlp_ratio), out_features=embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_heads_dim = self.head_dim * self.num_heads
        self.qkv = nn.Linear(in_features=embed_dim,
                             out_features=self.all_heads_dim * 3)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(self.all_heads_dim, embed_dim)
        self.softmax = nn.Softmax(-1)


    def transpose_multi_head(self, x):
        # x: [batch_size, num_patches, all_heads_dim]
        # x -> [batch_size, num_patches, num_heads, head_dim]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        # x -> [batch_size, num_heads, num_patches, head_dim]
        x.permute(0, 2, 1, 3)
        return x


    def forward(self, x):
        batch_size, num_patches, _ = x.shape
        # x:[batch_size, num_patches, embed_dim]
        # cut into three pieces along the last dim
        qkv = self.qkv(x).chunk(3, -1)
        # qkv: [batch_size, num_patches, all_heads_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)
        attn = torch.matmul(q, k.transpose(2, 3))
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)
        # attn: [batch_size, num_heads, num_patches, num_patches]

        out = torch.matmul(attn, v)
        # out: [batch_size, num_heads, num_patches, head_dim]
        out = out.permute(0, 2, 1, 3)
        out = out.reshape([batch_size, num_patches, -1])
        # out: [batch_size, num_patches, all_heads_dim]
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = Attention(embed_dim=embed_dim, num_heads=4)
        self.mlp = Mlp(embed_dim=embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = h + x
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x

class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(32, 1, 3, 96)
        layer_list = [Encoder(96) for i in range(5)]
        self.encoders = nn.Sequential(*layer_list)
        self.head = nn.Linear(96, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

def main():
    x = torch.randn(size=(4, 3, 32, 32))
    model = ViT(100)
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()
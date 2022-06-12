import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

with open('config_vit.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

class PatchEmbedding(nn.Module):
    
    def __init__(self, in_channels: int = parameters['in_channels'],
                 patch_size: int = parameters['patch_size'],
                 emb_size: int = parameters['emb_size'],
                 img_size: int = parameters['img_size']):
        
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),#geração dos patches
            Rearrange('b c (h) (w) -> b (h w) c'),
        )
        
        
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        
        
        #flat patches
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        
        
    def forward(self, x: Tensor):
        """
        debug: Parece que o modelo ta associando o mesmo cls_token para cada patch da imagem.
        """
        b, _, _, _ = x.shape
        
        x = self.projection(x)
        
        cls_tokens = repeat(self.cls_token, '() n c -> b n c', b=b)
        # prepend the cls token to the input
        
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = parameters['emb_size_multi_head_attention'],
                 num_heads: int = parameters['num_heads'],
                 dropout: float = parameters['dropout']):
        
        super().__init__()
        
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        """
        (h d) Representation division between dimension/num_heads
        task:colocar um raise na divisao num_heads por  embbeding gerado.
        h -> representa a quantidade heads geradas
        """
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        # matrix multiplication between queries and keys, a.k.a sum up over the last axis.
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        
        att = F.softmax(energy, dim=-1) / scaling
        
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 2, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = parameters['emb_size'],
                 drop_p: float = parameters['dropout'],
                 forward_expansion: int = parameters['expansion_embbeding'],
                 forward_drop_p: float = parameters['dropout'],
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 depth: int = parameters['depth'],
                 **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = parameters['emb_size'], n_classes: int = parameters['n_classes']):
        
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
        
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = parameters['in_channels'],
                patch_size: int = parameters['patch_size'],
                emb_size: int = parameters['emb_size'],
                img_size: int = parameters['img_size'],
                depth: int = parameters['depth'],
                n_classes: int = parameters['depth'],
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
import numpy as np
from LaMed.src.model.SwinUNetrClassification import SwinUNETR_Encoder
from LaMed.src.utils.dist_utils import gather_features
from torch.distributed import get_rank
# from coca_pytorch.coca_pytorch import CoCa
# Configuration class
import torch
from torch import einsum, nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.distributed as dist

from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# distributed

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_variable_batch(t):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    size = torch.tensor(t.shape[0], device = device, dtype = torch.long)
    sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
    dist.all_gather(sizes, size)

    sizes = torch.stack(sizes)
    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = 0)
    gathered_tensors = [torch.empty_like(padded_t, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')

    gathered_tensor = gathered_tensor[mask]
    sizes = sizes.tolist()

    return gathered_tensor, sizes

class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        assert dist.is_initialized() and dist.get_world_size() > 1
        x, batch_sizes = all_gather_variable_batch(x)
        ctx.batch_sizes = batch_sizes
        return x

    @staticmethod
    def backward(ctx, grads):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = 0)
        return grads_by_rank[rank]

all_gather = AllGather.apply


# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# to latents


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)

# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).to(torch.bfloat16) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.mask = None
        self.pos_emb = None

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n].to(device)

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.mask = mask
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)

        pos_emb = self.rotary_emb(n, device=device)
        self.pos_emb = pos_emb
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # extra attention mask - for masking out attention from text CLS token to padding

        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()
        # print(self.context_norm)
        # exit()
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        # print(x.shape, context.shape)
        # # print(self.context_norm)
        # exit()
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity
        # print(q.shape, k.shape)
        # exit()
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out

# transformer
class COCAConfig(PretrainedConfig):
    def __init__(
        self,
        language_model_name_or_path: str = "/research/d1/rshr/xgzhou/code/M3D/M3D/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        img_size: tuple = (128, 128, 128),
        hidden_size: int = 3584,
        vicunna_hidden_size: int = 4096,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.in_channels = in_channels
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.vicunna_hidden_size = vicunna_hidden_size
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

class CoCa(PreTrainedModel):
    def __init__(
        self,
        config,
        unimodal_depth=6,
        multimodal_depth=6,
        dim_latents=None,
        image_dim=None,
        num_img_queries=256,
        dim_head=64,
        heads=8,
        ff_mult=4,
        vision_encoder=None,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
        pad_id=0
    ):
        super().__init__(config)
        self.vision_encoder = SwinUNETR_Encoder(img_size=config.img_size, in_channels=config.in_channels, out_channels=4, feature_size=48).to(torch.bfloat16)
        dim = config.max_text_len
        self.dim = dim

        self.pad_id = pad_id
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # token embeddings
        num_tokens = config.vocab_size + 1
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.text_cls_token = nn.Parameter(torch.randn(dim))
        # attention pooling for image tokens
        # image_dim = config.img_dim
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(dim)
        self.text_cls_norm = LayerNorm(dim)

        # to latents

        dim_latents = default(dim_latents, dim)
        self.img_to_latents = EmbedToLatents(dim, dim_latents)
        self.text_to_latents = EmbedToLatents(dim, dim_latents)

        # contrastive learning temperature

        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # unimodal layers

        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
            )

        # multimodal layers

        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
            ]))

        # to logits

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        # self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)
        self.to_logits[-1].weight.data = self.token_emb.weight.data.clone()

        # whether in data parallel setting
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        self.count = 0

    def embed_text(self, text):
        batch, device = text.shape[0], text.device

        seq = text.shape[1]
        # Add this right before the failing line in embed_text method
        # print(f"Embedding table shape: {self.token_emb.weight.shape}")
        # print(f"Text indices min: {text.min().item()}, max: {text.max().item()}")
        # print(f"Text indices unique values: {torch.unique(text).tolist()}")
        # exit()
        text_tokens = self.token_emb(text)

        # append text cls tokens

        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

        # create specific mask for text cls token at the end
        # to prevent it from attending to padding

        cls_mask = rearrange(text!=self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # go through unimodal layers

        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)

        # get text cls token
        # print(text_tokens.shape)
        # exit()
        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        
        text_embeds = self.text_cls_norm(text_cls_tokens)
        return text_embeds, text_tokens

    def embed_image(self, images=None, image_tokens=None):
        # encode images into embeddings
        # with the vision_encoder passed in at init
        # it can also accept precomputed image tokens

        # assert not (exists(images) and exists(image_tokens))
        # print(f"self.vision_encoder: {self.vision_encoder}")
        # exit()
        if exists(images):
            assert exists(self.vision_encoder), 'vision_encoder must be passed in for automatic image encoding'
            image_tokens = self.vision_encoder(images)

        # attention pool image tokens
        image_tokens = image_tokens.mean(dim=2)
        image_tokens = rearrange(image_tokens, 'b c ... -> b c (...)')
        # print(image_tokens.shape)
        # exit()
        # image_tokens = image_tokens.view(image_tokens.size(0), image_tokens.size(1), -1)
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]

    def forward(
        self,
        images=None,
        input_ids=None,
        attention_mask=None,
        image_tokens=None,
        labels=None,
        return_loss=True,
        return_embeddings=False
    ):
        text = input_ids
        batch, device = text.shape[0], text.device
        # print(text[:, 0])
        # exit()
        # print(text[0,0], text[0,-2], text[0,-1])
        # exit()
        # if return_loss:
        #     text, labels = text[:, :-1], text[:, 1:]
        labels = text

        text_embeds, text_tokens = self.embed_text(text)
        # print(text_embeds.shape, text_tokens.shape)
        # print(text_embeds[0, 0])
        # print(text_tokens[0, :, 0])
        # # print(images.dtype)
        # exit()
        images = images.to(torch.bfloat16)
        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)

        # return embeddings if that is what the researcher wants

        if return_embeddings:
            return text_embeds, image_embeds

        # go through multimodal layers

        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
            text_tokens = cross_attn(text_tokens, image_tokens)

        # print(text_tokens.shape)
        # exit()    
        logits = self.to_logits(text_tokens)

        # if not return_loss:
        #     return logits

        # shorthand

        ce = F.cross_entropy

        # calculate caption loss (cross entropy loss)

        logits = rearrange(logits, 'b n c -> b c n')
        # print(logits.shape, labels.shape)
        # # print(labels[:, 0])
        # exit()
        caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight

        # embedding to latents

        text_latents = self.text_to_latents(text_embeds)
        image_latents = self.img_to_latents(image_embeds)

        # maybe distributed all gather
        global_batch = batch
        if self.is_distributed:
            latents = torch.stack((text_latents, image_latents), dim = 1)
            latents = all_gather(latents)
            text_latents, image_latents = latents.unbind(dim = 1)
            global_batch = text_latents.shape[0]
        # calculate contrastive loss

        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(global_batch, device=device)
        # print(sim.shape, contrastive_labels.shape)
        # print(sim, contrastive_labels)
        # exit()
        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight
        self.count += 1
        if self.count % 160 == 0 and dist.get_rank() == 0:
            print(f"Contrastive loss: {contrastive_loss}, Caption loss: {caption_loss}")
        ret = {
            "loss": caption_loss + contrastive_loss,
            "logits": sim,
        }
        # print(logits.shape, labels.shape)
        # exit()
        return ret



# class COCA(PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         vit = SwinUNETR_Encoder(img_size=config.img_size, in_channels=config.in_channels, out_channels=4, feature_size=48).to(torch.float32)
#         self.coca = CoCa(
#             dim = config.max_text_len,                     # model dimension
#             vision_encoder = vit,             # vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
#             image_dim = 1024,              # image embedding dimension, if not the same as model dimensions
#             num_tokens = config.vocab_size,            # number of text tokens
#             unimodal_depth = 6,            # depth of the unimodal transformer
#             multimodal_depth = 6,          # depth of the multimodal transformer
#             dim_head = 64,                 # dimension per attention head
#             heads = 8,                     # number of attention heads
#             caption_loss_weight = 1.,      # weight on the autoregressive caption loss
#             contrastive_loss_weight = 1.,  # weight on the contrastive loss between image and text CLS embeddings
#         ).cuda()


#     # train by giving CoCa your text and images with `return_loss = True`
#     def forward(self, images, texts, input_ids, attention_mask, labels, caption_labels=None, alpha=0.5, **kwargs):
#         # print(len(texts))
#         # print(texts[:, :-1])
#         # text = torch.randint(0, 20000, (4, 512)).cuda()
#         # print(input_ids.shape, attention_mask.shape)
#         # print(text.shape, images.shape)
#         # print(len(images))
#         # exit()
#         loss, logits = self.coca(
#             text = input_ids,
#             images = images,
#             return_loss = True  # set this to True to get the full caption + contrastive loss
#         )
#         ret = {
#             "loss": loss,
#             "logits": logits,
#         }
#         return ret

# Main COCA model
# class COCA(PreTrainedModel):
#     config_class = COCAConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.count = 0
#         self.vision_encoder = SwinUNETR_Encoder(img_size=config.img_size, in_channels=config.in_channels, out_channels=4, feature_size=48).to(torch.float32)
#         self.GAP = nn.Sequential(
#                 nn.GroupNorm(16, 3584), #TBD
#                 nn.ReLU(inplace=True),
#                 torch.nn.AdaptiveAvgPool3d((1,1,1)),
#                 # nn.Conv3d(3584, 256, kernel_size=1, stride=1, padding=0),
#                 nn.Flatten()
#         )

#         # self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)
#         # self.language_encoder = AutoModelForCausalLM.from_pretrained(config.language_model_name_or_path)
#         self.language_encoder = AutoModel.from_pretrained(config.language_model_name_or_path)
#         # self.language_encoder = AutoModel.from_pretrained("google-bert/bert-base-uncased")
#         self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size)
#         #self.mm_language_proj = nn.Linear(config.vicunna_hidden_size, config.hidden_size)
#         self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.local_loss = config.local_loss
#         self.gather_loss = config.gather_loss

#     def encode_image(self, image):
#         # print("In model:", image.dtype)
#         image_feats = self.vision_encoder(image) # ViT [1, 2049, 3584] SwinUNetr [1, 3584, 3, 3, 3]
#         image_feats = self.GAP(image_feats)

#         image_feats = self.mm_vision_proj(image_feats)
#         image_feats = F.normalize(image_feats, dim=-1)
        
#         return image_feats # [1, 3584]

#     def encode_text(self, input_id, attention_mask):
#         text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]
#         # print(text_feats.shape)
#         # exit()
#         text_feats = self.mm_language_proj(text_feats)
#         text_feats = F.normalize(text_feats, dim=-1)

#         return text_feats

#     def generate_text(self, image_features, input_ids, attention_mask):
#         """Generate text given image features and input tokens."""
#         # Encode image
#         image_features = self.encode_image(image_features)

#         # Encode partial text (for teacher-forcing during training)
#         text_features = self.language_encoder(input_ids, attention_mask=attention_mask)["last_hidden_state"]

#         # Decode using Transformer decoder
#         decoder_output = self.text_generator(tgt=text_features, memory=image_features.unsqueeze(1))
#         logits = self.lm_head(decoder_output)

#         return logits

#     def captioning_loss(self, logits, labels):
#         """Compute captioning loss using cross-entropy."""
#         # Shift labels to align with logits (ignore the first token)
#         shift_logits = logits[:, :-1, :].contiguous()
#         shift_labels = labels[:, 1:].contiguous()

#         # Flatten the logits and labels
#         loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#         return loss

#     def forward(self, images, input_ids, attention_mask, labels, caption_labels=None, alpha=0.5, **kwargs):
#         self.count += 1
#         # Contrastive Loss
#         image_features = self.encode_image(images)
#         text_features = self.encode_text(input_ids, attention_mask)[:, 0]

#         if self.gather_loss:
#             all_image_features, all_text_features = gather_features(image_features, text_features)
#             if self.local_loss:
#                 logits_per_image = self.logit_scale * image_features @ all_text_features.T
#                 logits_per_text = self.logit_scale * text_features @ all_image_features.T
#             else:
#                 # print("all_image_features", all_image_features.shape, "all_text_features", all_text_features.shape)
#                 # exit()
#                 logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = self.logit_scale * image_features @ text_features.T
#             logits_per_text = self.logit_scale * text_features @ image_features.T

#         contrastive_loss = (
#             F.cross_entropy(logits_per_image, labels) +
#             F.cross_entropy(logits_per_text, labels)
#         ) / 2

#         # Captioning Loss
#         if caption_labels is not None:
#             logits = self.generate_text(images, input_ids, attention_mask)
#             caption_loss = self.captioning_loss(logits, caption_labels)
#         else:
#             caption_loss = torch.tensor(0.0, device=images.device)
#         if self.count % 160 == 0 and get_rank() == 0:
#             print("contrastive_loss", contrastive_loss, "caption_loss", caption_loss)
#         # Total Loss
#         total_loss = alpha * contrastive_loss + (1 - alpha) * caption_loss

#         ret = {
#             "loss": total_loss,
#             "contrastive_loss": contrastive_loss,
#             "captioning_loss": caption_loss,
#             "logits": (logits_per_image + logits_per_text) / 2.0,
#         }

#         return ret


# def gather_features(image_features, text_features):
#     """Helper function to gather features across GPUs (dummy implementation)."""
#     # In a real distributed setting, this would use `torch.distributed.all_gather`.
#     return image_features, text_features
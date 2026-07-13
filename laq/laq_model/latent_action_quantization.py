from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack
from einops.layers.torch import Rearrange

from laq_model.attention import Transformer, ContinuousPositionBias
from laq_model.nsvq import NSVQ
from laq_model.heatmap_conditioning import apply_patch_heatmap, EgoMotionDirectionEmbedding
#from laq_model.costmap_loss import costmap_regularization_loss

def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class LatentActionQuantization(nn.Module):
    def __init__(
        self,
        *,
        dim,
        quant_dim,
        codebook_size,
        image_size,
        patch_size,
        spatial_depth,
        temporal_depth,
        dim_head = 64,
        heads = 8,
        channels = 3,
        attn_dropout = 0.,
        ff_dropout = 0.,
        code_seq_len = 1,
        heatmap_alpha = 1.0,
        heatmap_dir_alpha = 0.25,
        heatmap_abs_dir = False,       # Exp 2: use |fx|+|fy| instead of fx+fy
        heatmap_additive_dir = False,  # Exp 3: additive direction embedding
        #costmap_loss_weight = 0.0,
        #costmap_warmup_steps = 0,
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.code_seq_len = code_seq_len
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_dir_alpha = heatmap_dir_alpha
        self.heatmap_abs_dir = heatmap_abs_dir
        self.direction_embedding = EgoMotionDirectionEmbedding(dim) if heatmap_additive_dir else None
        #self.costmap_loss_weight = costmap_loss_weight
        #self.costmap_warmup_steps = costmap_warmup_steps
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )


        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        
        transformer_with_action_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
            has_cross_attn = True,
            dim_context = dim,
        )

        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)


        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            device='cuda',
            code_seq_len=code_seq_len,
            patch_size=patch_size,
            image_size=image_size
        )
            
            
        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **transformer_with_action_kwargs)
        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_height, p2 = patch_width)
        )


    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs, strict = False)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        pt = {k.replace('module.', '') if 'module.' in k else k: v for k, v in pt.items()}
        self.load_state_dict(pt)

    '''
    def _effective_costmap_loss_weight(self, step):
        """Holds costmap_loss_weight at exactly 0 for costmap_warmup_steps, then
        switches to the full target weight.

        This is a hard gate, not a gradual ramp -- an earlier linear-ramp
        version (nonzero from step 1) still collapsed the codebook
        identically to a constant weight. A controlled isolation test (same
        forward pass, costmap_regularization_loss's output manually
        .detach()-ed so it contributes literally zero gradient) recovered
        cleanly, while every version with ANY live gradient from this term
        -- regardless of loss formulation, weight magnitude (0.001-0.3), or
        gradient-clipping settings -- collapsed the codebook permanently.
        That isolates the cause to the gradient itself interfering with
        NSVQ's fragile early-training codebook formation specifically,
        not to overshooting some magnitude threshold. So: exactly zero
        gradient contribution (this hard gate) until the codebook has
        stabilized on its own (replace_unused_codebooks logs "Replaced 0"
        consistently by ~step 700-1000 in every healthy run), then the
        regularizer is introduced at full strength against an already-
        diverse, more robust representation. costmap_warmup_steps=0
        (default) disables the gate entirely -- constant weight from step 0,
        matching prior behavior.
        """
        if step < self.costmap_warmup_steps:
            return 0.0
        return self.costmap_loss_weight

    '''
    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]

        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        
        first_tokens = tokens[:, :1]
        last_tokens = tokens[:, 1:]
        
        return first_tokens, last_tokens

        

    def decode(
        self,
        tokens,
        actions,
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])


        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        actions = rearrange(actions, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape, context=actions)
        

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        rest_frames_tokens = tokens

        recon_video = self.to_pixels_first_frame(rest_frames_tokens)

        return recon_video
    

    def forward(
        self,
        video,
        step = 0,
        mask = None,
        heatmap = None,
        #costmap = None,
        return_recons_only = False,
        return_only_codebook_ids = False,
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]


        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb_first_frame(rest_frames)

        if exists(heatmap):
            first_frame_tokens = apply_patch_heatmap(first_frame_tokens, heatmap[:, :1], self.heatmap_alpha, self.heatmap_dir_alpha, self.heatmap_abs_dir)
            rest_frames_tokens = apply_patch_heatmap(rest_frames_tokens, heatmap[:, 1:], self.heatmap_alpha, self.heatmap_dir_alpha, self.heatmap_abs_dir)
            if exists(self.direction_embedding):  # Exp 3: additive direction on top of magnitude gate
                first_frame_tokens = self.direction_embedding(first_frame_tokens, heatmap[:, :1])
                rest_frames_tokens = self.direction_embedding(rest_frames_tokens, heatmap[:, 1:])

        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)

        shape = tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(tokens)

        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')


        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
        self.lookup_free_quantization = False
        vq_kwargs = dict(mask = vq_mask) if not self.lookup_free_quantization else dict()

        #effective_costmap_weight = self._effective_costmap_loss_weight(step)
        #use_costmap_loss = exists(costmap) and effective_costmap_weight > 0

        vq_out = self.vq(first_tokens, last_tokens, codebook_training_only = False)
        
        tokens, perplexity, codebook_usage, indices= vq_out
       

        num_unique_indices = indices.unique().size(0)

        if step > 500: # changed from step != 0 to step > 500 for more challenge
            if step % 10 == 0 and step < 100:
                self.vq.replace_unused_codebooks(10)
            elif step % 100 == 0 and step < 1000:
                self.vq.replace_unused_codebooks(100)
            elif step % 500 == 0 and step < 5000:
                self.vq.replace_unused_codebooks(500)
        

        if return_only_codebook_ids:
            return indices
        
        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            ## error
            print("code_seq_len should be square number or defined as 2")
            return
        
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        concat_tokens = first_frame_tokens.detach() # + tokens
        recon_video = self.decode(concat_tokens, tokens)

        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 

        if return_recons_only:
            return returned_recon

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video, recon_video)

        #if use_costmap_loss:
        #    costmap_agg = costmap.amax(dim = 1)  # (b, 2, h, w) -> (b, h, w); risk from either frame counts
        #    costmap_loss = costmap_regularization_loss(continuous_latent, costmap_agg)
        #else:
        #    costmap_loss = recon_loss.new_zeros(())

        #loss = recon_loss + effective_costmap_weight * costmap_loss
        loss = recon_loss

        #aux_logs = {'recon_loss': recon_loss.item(), 'costmap_loss': costmap_loss.item()}
        aux_logs = {'recon_loss': recon_loss.item()}

        return loss, num_unique_indices, aux_logs
        

    def inference(
        self,
        video,
        step = 0,
        mask = None,
        heatmap = None,
        return_only_codebook_ids=False,
        user_action_token_num=None
    ):

        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb_first_frame(rest_frames)

        if exists(heatmap):
            first_frame_tokens = apply_patch_heatmap(first_frame_tokens, heatmap[:, :1], self.heatmap_alpha, self.heatmap_dir_alpha, self.heatmap_abs_dir)
            rest_frames_tokens = apply_patch_heatmap(rest_frames_tokens, heatmap[:, 1:], self.heatmap_alpha, self.heatmap_dir_alpha, self.heatmap_abs_dir)
            if exists(self.direction_embedding):  # Exp 3: additive direction on top of magnitude gate
                first_frame_tokens = self.direction_embedding(first_frame_tokens, heatmap[:, :1])
                rest_frames_tokens = self.direction_embedding(rest_frames_tokens, heatmap[:, 1:])

        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)


        shape = tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(tokens)

        # quantize
        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')

        if user_action_token_num is not None:
            tokens, indices = self.vq.inference(first_tokens, last_tokens, user_action_token_num=user_action_token_num)
        else:
            tokens, indices = self.vq.inference(first_tokens, last_tokens)

        
    
        if return_only_codebook_ids:
            return indices

        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            print("code_seq_len should be square number or defined as 2")
            return
        

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        concat_tokens = first_frame_tokens #.detach() #+ tokens
        recon_video = self.decode(concat_tokens, actions=tokens)
        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 
        
        return returned_recon


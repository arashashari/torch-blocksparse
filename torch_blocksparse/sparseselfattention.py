import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple
import torch_blocksparse
import sys
import random

class SparseSelfAttention(nn.Module):
    ops = dict()

    # add to cache
    def get_ops(self, H, L):
        import sys
        if L not in SparseSelfAttention.ops:
            sparse_dot_sdd_nt = torch_blocksparse.MatMul(self.sparsity_config.layout,
                    self.sparsity_config.block,
                    'sdd',
                    trans_a=False,
                    trans_b=True)

            sparse_dot_dsd_nn = torch_blocksparse.MatMul(self.sparsity_configlayout,
                    self.sparsity_config.block,
                    'dsd',
                    trans_a=False,
                    trans_b=False)

            sparse_softmax = torch_blocksparse.Softmax(self.sparsity_config.layout, self.sparsity_config.block)

            SparseSelfAttention.ops[L] = (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax)
        return SparseSelfAttention.ops[L]

    # constructor
    def __init__(self, sparsity_config=torch_blocksparse.SparsityConfig(), key_padding_mask_mode='add', attn_mask_mode='mul'):
        super().__init__()

        # sparsity information
        self.sparsity_config = sparsity_config
       
        # mask modes
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    # forward pass
    def forward(self, query, key, value, rpe=None, key_padding_mask=None, attn_mask=None):
        bsz, num_heads, tgt_len, head_dim = query.size()
        
        # transpose back key if it is already transposed
        key = self.transpose_key_for_scores(key, tgt_len)

        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')

        # squeeze key_padding_mask if it is given
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype, key_padding_mask, is_key_padding_mask=True)


        # squeeze attn_mask if it is given
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)

        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(num_heads, tgt_len)

        scaling = float(head_dim) ** -0.5

        # attention scores
        attn_output_weights = sparse_dot_sdd_nt(query, key)
        attn_output_weights = sparse_softmax(attn_output_weights, scale=scaling, rpe=rpe, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)

        # outputs
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output
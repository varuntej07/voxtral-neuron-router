"""Model-side workarounds for Neuron compiler bugs, applied before torch_neuronx.trace.

patch_encoder_positions
    VoxtralEncoder.forward does `gelu(conv2(...)).permute(0, 2, 1) + embed_positions.weight`.
    Traced with torch_neuronx.trace (neuronx-cc 2.26.6360.0), an add of a module weight or
    buffer right after a permute comes out as if the weight's [T, C] memory were [C, T]:
    scripts/repro_permute_add.py reproduces it in five lines, cosine 0.50 at the encoder
    shape, and the same add with the tensor passed as an input is exact. In the full encoder
    it showed up as cosine 0.0975 at depth 0 and 0.372 at full depth.

    The patch adds the positions before the permute, from a buffer stored pre-transposed
    as [C, T]. Same arithmetic, in the same dtypes, so CPU output is bit-identical, and no
    weight is added after a permute any more.
"""

import types


def patch_encoder_positions(tower):
    """Replace tower.forward in place with the permute-free-add version. Returns tower."""
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    # [T, C] -> [1, C, T], kept in the embedding's own dtype (fp32 in Voxtral), so the add
    # promotes exactly as the original (bf16 + fp32 -> fp32, then back to bf16).
    pos_ct = tower.embed_positions.weight.detach().T.contiguous().unsqueeze(0)
    tower.register_buffer("pos_ct", pos_ct, persistent=False)

    def forward(self, input_features, attention_mask=None, **kwargs):
        x = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        h = torch.nn.functional.gelu(self.conv1(x))
        h = torch.nn.functional.gelu(self.conv2(h))
        h = (h + self.pos_ct).to(h.dtype).permute(0, 2, 1)  # add first, then permute
        h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, layer_head_mask=None)[0]
        return BaseModelOutput(last_hidden_state=self.layer_norm(h))

    tower.forward = types.MethodType(forward, tower)
    return tower

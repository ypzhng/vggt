import os
import typing
import logging
import torch
import ipdb
import transformers
from . import utils
# from typing import Callable, Any, Optional

# New: import VGGT
try:
    from vggt.models.vggt import VGGT as VGGT_MODEL
except Exception:
    VGGT_MODEL = None  # Allow file to import even if VGGT isn't installed

# Hugging Face model/type aliases
OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer

# Optional: resolve VGGT block types for finer-grained checks during capture
# We won’t import internal blocks by path to keep this robust; we’ll match by attribute names at runtime.


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif VGGT_MODEL is not None and isinstance(model, VGGT_MODEL):
        return VGGT_MODEL
    else:
        raise ValueError(f'Unknown model type {type(model)}')


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def get_rope_function_name(model):
    # LLAMA uses apply_rotary_pos_emb (per HF Llama)
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    # VGGT Attention modules include RotaryPositionEmbedding2D (rope attribute),
    # but there is no single exported function name comparable to HF Llama.
    # Return a sentinel to indicate 2D RoPE presence; callers should handle if needed.
    if VGGT_MODEL is not None and isinstance(model, VGGT_MODEL):
        return "rotary_position_embedding_2d"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    if VGGT_MODEL is not None and isinstance(model, VGGT_MODEL):
        # For VGGT we expose a flat list of transformer-like blocks that users commonly introspect.
        # Priority: aggregator.patch_embed.blocks (NestedTensorBlock), aggregator.frame_blocks (Block),
        # aggregator.global_blocks (Block), camera_head.trunk (Sequential of Block)
        layers = []

        # aggregator.patch_embed.blocks: DinoVisionTransformer -> blocks: ModuleList of NestedTensorBlock
        agg = getattr(model, "aggregator", None)
        if agg is not None:
            # patch path
            # patch_vit = getattr(agg, "patch_embed", None)
            # if patch_vit is not None:
            #     pe_blocks = getattr(patch_vit, "blocks", None)
            #     if pe_blocks is not None:
            #         layers.extend(list(pe_blocks))  # NestedTensorBlock

            # frame blocks
            frame_blocks = getattr(agg, "frame_blocks", None)
            if frame_blocks is not None:
                layers.extend(list(frame_blocks))  # Block

            # global blocks
            global_blocks = getattr(agg, "global_blocks", None)
            if global_blocks is not None:
                layers.extend(list(global_blocks))  # Block

        # camera_head.trunk is Sequential of Block(s)
        # camera_head = getattr(model, "camera_head", None)
        # if camera_head is not None:
        #     trunk = getattr(camera_head, "trunk", None)
        #     if trunk is not None:
        #         layers.extend(list(trunk))  # Block

        # Return only modules that look like transformer blocks (have attn and mlp)
        pruned = []
        for l in layers:
            if hasattr(l, "attn") and hasattr(l, "mlp"):
                pruned.append(l)
        return pruned

    raise NotImplementedError

def _unwrap(m):
    return getattr(m, 'module', m)

def get_model_sizes(model):
    # Original path for HuggingFace-style models
    if hasattr(model, "config"):
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        intermediate_size = getattr(model.config, "intermediate_size", None)
        return hidden_size, num_heads, intermediate_size

    # Minimal VGGT support: read from Aggregator's blocks
    agg = getattr(model, "aggregator", model)  # allow passing either VGGT or Aggregator
    blocks = None
    # Prefer frame_blocks, then global_blocks
    if hasattr(agg, "frame_blocks") and len(agg.frame_blocks) > 0:
        blocks = agg.frame_blocks
    elif hasattr(agg, "global_blocks") and len(agg.global_blocks) > 0:
        blocks = agg.global_blocks
    else:
        raise ValueError("Could not find frame_blocks/global_blocks on the model for VGGT.")

    blk = blocks[0]

    # hidden_size from attn.proj Linear in_features (unwrap ActQuantWrapper if present)
    attn = getattr(blk, "attn", None)
    proj = getattr(attn, "proj", None) if attn is not None else None
    hidden_size = None
    if proj is not None:
        lin = _unwrap(proj)
        if isinstance(lin, torch.nn.Linear):
            hidden_size = lin.in_features

    # Fallback via mlp.fc2 out_features
    if hidden_size is None and hasattr(blk, "mlp") and hasattr(blk.mlp, "fc2"):
        lin = _unwrap(blk.mlp.fc2)
        if isinstance(lin, torch.nn.Linear):
            hidden_size = lin.out_features

    # num_heads: prefer block.num_heads, else attn.num_heads
    num_heads = getattr(blk, "num_heads", None)
    if num_heads is None and attn is not None:
        num_heads = getattr(attn, "num_heads", None)

    # intermediate_size from mlp.fc1 out_features
    intermediate_size = None
    if hasattr(blk, "mlp") and hasattr(blk.mlp, "fc1"):
        fc1_lin = _unwrap(blk.mlp.fc1)
        if isinstance(fc1_lin, torch.nn.Linear):
            intermediate_size = fc1_lin.out_features

    if hidden_size is None or num_heads is None:
        raise ValueError("Could not determine hidden_size/num_heads from VGGT Aggregator blocks.")

    return hidden_size, num_heads, intermediate_size


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", use_auth_token=hf_token, low_cpu_mem_usage=True
    )
    model.seqlen = 2048
    logging.info("---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen))
    return model


def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", low_cpu_mem_usage=True
    )
    model.seqlen = model.config.max_position_embeddings
    logging.info("---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen))
    return model


def get_vggt(model_name: str, ckpt_path: str | None = None, **kwargs):
    """
    Load VGGT. As it is not from transformers, we assume the caller provides either:
      - a model name resolvable by VGGT code, or
      - a ckpt_path to load state_dict from.

    kwargs are passed to VGGT init if needed by your project.
    """
    if VGGT_MODEL is None:
        raise ImportError("VGGT is not available. Please `from vggt.models.vggt import VGGT` in your environment.")
    # Construction varies across repos. If your VGGT requires from_pretrained-like API,
    # adapt this loader accordingly. Here we assume init + optional load_state_dict.
    model = VGGT_MODEL(**kwargs) if kwargs else VGGT_MODEL()
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")
        # Accept both full checkpoint and plain state_dict
        state_dict = sd.get("state_dict", sd)
        model.load_state_dict(state_dict, strict=False)
    # Most vision transformers operate on images; there is no seqlen concept.
    model.seqlen = None
    logging.info("---> Loading VGGT {} (ckpt: {})".format(model_name, ckpt_path))
    return model


def get_model(model_name, hf_token=None, **kwargs):
    """
    Extended to support VGGT. Usage examples:
      - model_name contains 'llama' or 'opt' -> as before.
      - model_name contains 'vggt' -> call get_vggt. You may pass ckpt_path=... via kwargs.
    """
    lname = model_name.lower()
    if "llama" in lname:
        return get_llama(model_name, hf_token)
    elif "opt" in lname:
        return get_opt(model_name)
    elif "vggt" in lname:
        return get_vggt(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model {model_name}")


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif VGGT_MODEL is not None and isinstance(model, VGGT_MODEL):
        model_type = VGGT_MODEL
    else:
        raise ValueError(f"Unknown model type {type(model)}")
    return model_type


def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif VGGT_MODEL is not None and model_type == VGGT_MODEL:
        # For VGGT, the closest notion to input "embeddings" is the patch embedding conv in the image trunk.
        # Expose both:
        # - aggregator.patch_embed.patch_embed.proj (Conv2d)
        # - optional token/positional embeddings if present (none here)
        embs = []
        agg = getattr(model, "aggregator", None)
        if agg is not None:
            patch_vit = getattr(agg, "patch_embed", None)
            if patch_vit is not None:
                pe = getattr(patch_vit, "patch_embed", None)
                if pe is not None:
                    proj = getattr(pe, "proj", None)
                    if proj is not None:
                        embs.append(proj)
        return embs
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif VGGT_MODEL is not None and model_type == VGGT_MODEL:
        return get_layers(model)
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif VGGT_MODEL is not None and model_type == VGGT_MODEL:
        # VGGT is a vision model; no LM head.
        return None
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif VGGT_MODEL is not None and model_type == VGGT_MODEL:
        # For VGGT, the final normalization before heads depends on the pathway.
        # We expose aggregator.patch_embed.norm (Identity in the provided printout) and
        # aggregator.{frame,global} blocks' final norm would be inside each Block.
        # There isn't a single global "pre_head" norm; return None and let caller handle.
        pre_head_layernorm = None
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return pre_head_layernorm


def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    elif VGGT_MODEL is not None and model_type == VGGT_MODEL:
        # For VGGT Blocks, MLP typically uses fc1 expanding to hidden_mult (e.g., 4x).
        # The bottleneck is the hidden (fc1 out_features) of the common Block width.
        # We cannot read a single config; return a representative size by peeking at first block with mlp.fc1.
        layers = get_layers(model)
        for l in layers:
            mlp = getattr(l, "mlp", None)
            if mlp is not None and hasattr(mlp, "fc1") and hasattr(mlp.fc1, "out_features"):
                return mlp.fc1.out_features
        return None
    else:
        raise ValueError(f"Unknown model type {model_type}")


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name) if name.isdigit() else name)
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)
            
# def replace_modules_selective(
#     root: torch.nn.Module,
#     type_to_replace: type,
#     new_module_factory: typing.Callable[[torch.nn.Module, str, typing.Optional[torch.nn.Module]], torch.nn.Module],
#     should_replace: typing.Callable[[torch.nn.Module, str, typing.Optional[torch.nn.Module]], bool],
# ) -> None:
#     """
#     Depth-first replacement with a filter. Only replaces modules of type_to_replace
#     for which should_replace(parent, name, module) returns True.

#     Args:
#       root: root module to traverse
#       type_to_replace: class to match (e.g., torch.nn.LayerNorm)
#       new_module_factory: function (parent, name, module) -> replacement module
#       should_replace: predicate (parent, name, module) -> bool
#     """
#     for name, module in list(root.named_children()):
#         # Recurse first so we can still visit grandchildren if we don't replace this node
#         replace_modules_selective(module, type_to_replace, new_module_factory, should_replace)

#         if isinstance(module, type_to_replace) and should_replace(root, name, module):
#             new_module = new_module_factory(root, name, module)
#             setattr(root, name, new_module)

def replace_modules_selective(
    root: torch.nn.Module,
    type_to_replace: type,
    new_module_factory: typing.Callable[[torch.nn.Module, str, torch.nn.Module], torch.nn.Module],
    should_replace: typing.Callable[[torch.nn.Module, str, str, torch.nn.Module], bool],
    qualified_name: str = "",
) -> None:
    """
    Depth-first replacement with a filter. Only replaces modules of type_to_replace
    for which should_replace(parent, parent_qualified_name, child_name, child_module) returns True.

    Args:
      root: current parent module in traversal
      type_to_replace: class to match (e.g., nn.LayerNorm)
      new_module_factory: function (parent, child_name, child_module) -> replacement module
      should_replace: predicate (parent, parent_qualified_name, child_name, child_module) -> bool
      qualified_name: fully qualified path of 'root' within the whole model
    """
    # We must list children first because we might replace attributes while iterating
    for child_name, child_module in list(root.named_children()):
        child_qn = f"{qualified_name}.{child_name}" if qualified_name else child_name

        # Recurse before replacement so grandchildren are visited regardless
        replace_modules_selective(
            child_module,
            type_to_replace,
            new_module_factory,
            should_replace,
            qualified_name=child_qn,
        )

        # Decide replacement on the child sitting under 'root'
        if isinstance(child_module, type_to_replace) and should_replace(root, qualified_name, child_name, child_module):
            new_module = new_module_factory(root, child_name, child_module)
            setattr(root, child_name, new_module)

class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
    
class LNNoAffine(torch.nn.Module):
    """
    LayerNorm without affine parameters, used to replace LayerNorm after
    folding gamma/beta into adjacent Linear layers.

    Equivalent to torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False),
    but implemented explicitly to avoid relying on newer PyTorch features.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over the last len(normalized_shape) dims, like torch LayerNorm
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat.to(input_dtype)

# class LNNoAffine(torch.nn.Module):
#     def __init__(self, normalized_shape, eps=1e-5, upcast_fp16=True):
#         super().__init__()
#         if isinstance(normalized_shape, int):
#             normalized_shape = (normalized_shape,)
#         self.norm = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)
#         self.upcast_fp16 = upcast_fp16

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         orig_dtype = x.dtype
#         if self.upcast_fp16 and x.dtype == torch.float16:
#             x = x.to(torch.float32)
#         y = self.norm(x)
#         return y.to(orig_dtype)


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, "layer_io", f"{args.layer_idx:03d}.pt")


def _register_hook(module, hook):
    if module is None:
        return None
    return module.register_forward_hook(hook)


def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            "k_proj": [],  # q_proj, v_proj has the same input as k_proj
            "o_proj": [],
            "gate_proj": [],  # up_proj has the same input as gate_proj
            "down_proj": [],
        }

        captured_outputs = {
            "v_proj": [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            "k_proj": [],  # q_proj, v_proj has the same input as k_proj
            "out_proj": [],
            "fc1": [],
            "fc2": [],
        }
        captured_outputs = {
            "v_proj": [],
        }
        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif VGGT_MODEL is not None and model_type == VGGT_MODEL:
        # VGGT Block anatomy (from the provided printout):
        # - attn with (qkv: Linear, proj: Linear) and drops; may have rope, q_norm, k_norm
        # - mlp with (fc1: Linear, fc2: Linear)
        # We will capture:
        #   inputs: qkv (shared input for q,k,v), proj, fc1, fc2
        #   outputs: qkv (to inspect split), proj
        captured_inputs = {
            "qkv": [],    # shared input to qkv
            "proj": [],   # input to output projection
            "fc1": [],
            "fc2": [],
        }
        captured_outputs = {
            "qkv": [],
            "proj": [],
        }

        attn = getattr(layer, "attn", None)
        mlp = getattr(layer, "mlp", None)

        # Some layers (e.g., NestedTensorBlock) still have attn/mlp with the same names
        qkv_mod = getattr(attn, "qkv", None) if attn is not None else None
        proj_mod = getattr(attn, "proj", None) if attn is not None else None
        fc1_mod = getattr(mlp, "fc1", None) if mlp is not None else None
        fc2_mod = getattr(mlp, "fc2", None) if mlp is not None else None

        if qkv_mod is not None:
            handles.append(_register_hook(qkv_mod, hook_factory("qkv", captured_inputs, True)))
            handles.append(_register_hook(qkv_mod, hook_factory("qkv", captured_outputs, False)))
        if proj_mod is not None:
            handles.append(_register_hook(proj_mod, hook_factory("proj", captured_inputs, True)))
            handles.append(_register_hook(proj_mod, hook_factory("proj", captured_outputs, False)))
        if fc1_mod is not None:
            handles.append(_register_hook(fc1_mod, hook_factory("fc1", captured_inputs, True)))
        if fc2_mod is not None:
            handles.append(_register_hook(fc2_mod, hook_factory("fc2", captured_inputs, True)))

    else:
        raise ValueError(f"Unknown model type {model_type}")

    # Process each sequence in the batch one by one to avoid OOM.
    # For VGGT, "sequence" is whatever layer_input represents (e.g., tokens/features).
    for seq_idx in range(layer_input.shape[0]):
        seq = layer_input[seq_idx : seq_idx + 1].to(utils.DEV)
        layer(seq)

    # Concatenate accumulated inputs/outputs across the batch.
    for module_name in list(locals().get("captured_inputs", {}).keys()):
        if len(captured_inputs[module_name]) > 0:
            captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in list(locals().get("captured_outputs", {}).keys()):
        if len(captured_outputs[module_name]) > 0:
            captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        if h is not None:
            h.remove()

    return {
        "input": locals().get("captured_inputs", {}),
        "output": locals().get("captured_outputs", {}),
    }
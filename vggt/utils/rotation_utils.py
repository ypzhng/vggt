from vggt.utils import model_utils
import torch
import typing
from . import utils
import transformers
import tqdm, math
import ipdb
# from vggt.utils import quant_utils
from vggt.utils.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform

# def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
#     """
#     fuse the linear operations in Layernorm into the adjacent linear blocks.
#     """
#     for linear in linear_layers:
#         linear_dtype = linear.weight.dtype

#         # Calculating new weight and bias
#         W_ = linear.weight.data.double()
#         linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

#         if hasattr(layernorm, 'bias'):
#             if linear.bias is None:
#                 linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
#             linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
#             linear.bias.data = linear.bias.data.to(linear_dtype)

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    Fuse the affine part of LayerNorm into following Linear layers:
      y = LN(x) = (x - mu) / sigma * gamma + beta
      Linear(y) = W y + b
    We fold only the affine scaling/shift (gamma/beta) here:
      W' = W * gamma
      b' = b + W * beta
    Note: We do not fold the input-dependent normalization ((x - mu) / sigma),
    which cannot be absorbed statically. This matches the original behavior.
    """
    if layernorm is None:
        return
    # Some model variants might use Identity for norm; skip if no weight
    if not hasattr(layernorm, "weight"):
        return

    gamma = layernorm.weight.data.double()  # [hidden]
    beta = layernorm.bias.data.double() if hasattr(layernorm, "bias") and layernorm.bias is not None else None

    for linear in linear_layers:
        if linear is None or not isinstance(linear, torch.nn.Linear):
            continue

        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()  # [out, hidden]

        # Scale columns of W by gamma (broadcast over hidden dim)
        # Equivalent to W @ diag(gamma)
        W = W_ * gamma  # broadcasting over last dim (hidden)
        linear.weight.data = W.to(linear_dtype)

        # Bias folding with beta
        if beta is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            b = linear.bias.data.double()
            b = b.double() + torch.matmul(W_, beta.double())  # original W 
            linear.bias.data = b.to(linear_dtype)

            
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def is_vggt_core_block(module: torch.nn.Module) -> bool:
    """
    Heuristic: VGGT core transformer-like blocks expose:
      - attributes 'attn' and 'mlp'
      - attn has a Linear named 'qkv' (VGGT Attention(qkv, proj, ...))
    TrackHead AttnBlock uses nn.MultiheadAttention and has no 'qkv' Linear, so this returns False.
    """
    if not (hasattr(module, "attn") and hasattr(module, "mlp")):
        return False
    attn = getattr(module, "attn", None)
    if attn is None:
        return False
    qkv = getattr(attn, "qkv", None)
    return isinstance(qkv, torch.nn.Linear)


def vggt_fused_ln_should_replace(parent: torch.nn.Module, name: str, module: torch.nn.Module) -> bool:
    """
    Replace ONLY LayerNorms that:
      - Live on a VGGT core block (is_vggt_core_block(parent) is True), and
      - Are named 'norm1' or 'norm2'.

    This skips:
      - LayerNorms inside attention (e.g., q_norm, k_norm) because their parent is 'attn', not the block.
      - LayerNorms in TrackHead/AttnBlock time_blocks, since is_vggt_core_block(parent) will be False.
      - Any other LayerNorms we did not fuse into.
    """
    if not isinstance(module, torch.nn.LayerNorm):
        return False
    if not is_vggt_core_block(parent):
        return False
    return name in ("norm1", "norm2")
            
# def fuse_layer_norms(model):
    
#     model_type = model_utils.get_model_type(model)
    
#     kwargs = {'model': model, 'model_type': model_type}
    
#     # Embedding fusion
#     for W in model_utils.get_embeddings(**kwargs):
#         W_ = W.weight.data.double()
#         W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
#     layers = model_utils.get_transformer_layers(**kwargs)
    
#     # Fuse the linear operations in Layernorm into the adjacent linear blocks.
#     for layer in layers:
        
#         # fuse the input layernorms into the linear layers
#         if model_type == model_utils.LLAMA_MODEL:
#             fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
#             fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
#         elif model_type == model_utils.OPT_MODEL:
#             fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
#             fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
#         else:
#             raise ValueError(f'Unknown model type {model_type}')
            
            
    
#         if model_type == model_utils.OPT_MODEL:
#             bake_mean_into_linear(layer.self_attn.out_proj)
#             bake_mean_into_linear(layer.fc2)
                    
    
#     fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
#     model_utils.replace_modules(
#         model,
#         transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
#         lambda _: model_utils.RMSN(model.config.hidden_size),
#         replace_layers=False,
#     )

def to_rmsn_from_layernorm_factory(orig_ln: torch.nn.LayerNorm):
    """
    Build an RMSN module whose mean_dim matches the LayerNorm's normalized shape.
    LayerNorm.normalized_shape is a tuple like (hidden_size,). We take the product.
    """
    if not isinstance(orig_ln, torch.nn.LayerNorm):
        raise TypeError(f"Expected LayerNorm, got {type(orig_ln)}")
    # Compute feature dimension LN normalizes over
    if isinstance(orig_ln.normalized_shape, (tuple, list)):
        mean_dim = 1
        for d in orig_ln.normalized_shape:
            mean_dim *= int(d)
    else:
        mean_dim = int(orig_ln.normalized_shape)
    # Keep eps consistent with the original LN to minimize numerical drift
    return model_utils.RMSN(mean_dim=mean_dim, eps=orig_ln.eps)

def ln_no_affine_factory(_parent: torch.nn.Module, _name: str, orig_ln: torch.nn.LayerNorm) -> torch.nn.Module:
    return model_utils.LNNoAffine(normalized_shape=orig_ln.normalized_shape, eps=orig_ln.eps)


def replace_post_fusion_norms_with_rmsn(model):
    """
    After fusing LN affine parameters into adjacent Linear layers, replace:
      - LLaMA RMSNorm with RMSN (already in your code)
      - OPT/VGGT LayerNorm with RMSN of appropriate mean_dim
    This ensures we don’t double-apply affine scaling and preserves correctness.
    """
    model_type = model_utils.get_model_type(model)

    if model_type == model_utils.LLAMA_MODEL:
        # Keep your original LLaMA branch (replace LlamaRMSNorm -> RMSN)
        model_utils.replace_modules(
            model,
            transformers.models.llama.modeling_llama.LlamaRMSNorm,
            lambda _: model_utils.RMSN(model.config.hidden_size),
            replace_layers=False,
        )
        # LLaMA has no torch.nn.LayerNorm in standard configs; nothing else to do.

    elif model_type == model_utils.OPT_MODEL:
        # OPT uses torch.nn.LayerNorm. Replace all with RMSN using each LN's normalized_shape.
        model_utils.replace_modules(
            model,
            torch.nn.LayerNorm,
            ln_no_affine_factory,
            replace_layers=False,
        )

    elif model_utils.VGGT_MODEL is not None and isinstance(model, model_utils.VGGT_MODEL):
        # VGGT widely uses torch.nn.LayerNorm in blocks and heads.
        # Replace every LayerNorm with RMSN matched by normalized_shape.
        # ipdb.set_trace()
        # model_utils.replace_modules(
        #     model,
        #     torch.nn.LayerNorm,
        #     to_ln_no_affine_factory,
        #     replace_layers=False,
        # )
        model_utils.replace_modules_selective(
            model,
            torch.nn.LayerNorm,
            new_module_factory=ln_no_affine_factory,
            should_replace=vggt_fused_ln_should_replace,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def fuse_layer_norms(model):
    model_type = model_utils.get_model_type(model)
    kwargs = {'model': model, 'model_type': model_type}

    # Embedding fusion/centering
    # - LLaMA/OPT: center embeddings as before.
    # - VGGT: center Conv2d weights along out_channels (per-filter mean over in_channels*kH*kW).
    for W in model_utils.get_embeddings(**kwargs):
        if isinstance(W, torch.nn.Embedding):
            ipdb.set_trace
            W_ = W.weight.data.double()
            W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        elif isinstance(W, torch.nn.Conv2d):#leave this part as we do not intend to rotate conv layer
            # W_ = W.weight.data.double()  # [out_c, in_c, kH, kW]
            # mean = W_.view(W_.shape[0], -1).mean(dim=1, keepdim=True)  # per out_channel
            # mean = mean.view(-1, 1, 1, 1)
            # W.weight.data = (W_ - mean).to(W.weight.data.dtype)
            # Bias centering usually not required; leave bias unchanged.
            pass
        else:
            # Any other module type is unexpected here; skip
            pass

    layers = model_utils.get_transformer_layers(**kwargs)

    for layer in layers:
        if model_type == model_utils.LLAMA_MODEL:
            # LLaMA fusions (unchanged)
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])

        elif model_type == model_utils.OPT_MODEL:
            # OPT fusions (unchanged)
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])

        elif model_utils.VGGT_MODEL is not None and isinstance(model, model_utils.VGGT_MODEL):
            # VGGT fusions:
            # Each block has:
            #   norm1 -> attn(qkv, proj)
            #   norm2 -> mlp(fc1, fc2)
            # Some variants use Identity; fuse only when LayerNorm present.
            norm1 = getattr(layer, "norm1", None)
            norm2 = getattr(layer, "norm2", None)
            attn = getattr(layer, "attn", None)
            mlp = getattr(layer, "mlp", None)

            # Fuse norm1 into attention projections
            if isinstance(norm1, torch.nn.LayerNorm) and attn is not None:
                qkv = getattr(attn, "qkv", None)
                if isinstance(qkv, torch.nn.Linear):
                    fuse_ln_linear(norm1, [qkv])

            # Fuse norm2 into MLP layers
            if isinstance(norm2, torch.nn.LayerNorm) and mlp is not None:
                fc1 = getattr(mlp, "fc1", None)
                if isinstance(fc1, torch.nn.Linear):
                    fuse_ln_linear(norm2, [fc1])

        else:
            raise ValueError(f'Unknown model type {model_type}')

        # OPT-specific baking (unchanged)
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)

    # Pre-head fusion:
    # - LLaMA/OPT: fuse final norm into LM head (if present)
    # - VGGT: get_lm_head returns None; skip safely
    pre_head_ln = model_utils.get_pre_head_layernorm(**kwargs)
    lm_head = model_utils.get_lm_head(**kwargs)
    if isinstance(pre_head_ln, (torch.nn.LayerNorm, transformers.models.llama.modeling_llama.LlamaRMSNorm)) and lm_head is not None:
        fuse_ln_linear(pre_head_ln, [lm_head])
        
    replace_post_fusion_norms_with_rmsn(model)
    

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    

def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    
# def rotate_attention_inputs(layer, Q, model_type) -> None:
#     # Rotate the WQ, WK and WV matrices of the self-attention layer.
#     for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
#         dtype = W.weight.dtype
#         W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
#         W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs(layer, Q, model_type) -> None:
    """
    Rotate the attention input projections by right-multiplying their weights with Q.
    - LLaMA-style: layer.self_attn.{q_proj,k_proj,v_proj}
    - VGGT-style:  layer.attn.qkv packed as [Q; K; V] along out_features

    This keeps the original dtype/device behavior (compute in float64, write back).
    """

    def _rot_weight_inplace(W_lin, Q_mat):
        dtype = W_lin.weight.dtype
        dev = W_lin.weight.device
        W64 = W_lin.weight.data.to(device=dev, dtype=torch.float64)
        Q64 = Q_mat.to(device=dev, dtype=torch.float64)
        W_lin.weight.data = (W64 @ Q64).to(device=dev, dtype=dtype)

    # Case 1: LLaMA-style separate projections
    if hasattr(layer, "self_attn") and all(
        hasattr(layer.self_attn, name) for name in ("q_proj", "k_proj", "v_proj")
    ):
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            _rot_weight_inplace(W, Q)
        return

    # Case 2: VGGT-style packed qkv under layer.attn.qkv or layer.qkv
    attn = getattr(layer, "attn", layer)
    qkv = getattr(attn, "qkv", None)
    if isinstance(qkv, torch.nn.Linear):
        W = qkv.weight
        out_features, in_features = W.shape
        if out_features % 3 != 0:
            raise ValueError(f"qkv weight out_features ({out_features}) not divisible by 3")
        hidden = in_features
        if out_features != 3 * hidden:
            raise ValueError(f"Expected qkv.out_features == 3*in_features, got {out_features} vs 3*{hidden}")

        # Compute in float64 on the same device
        dev = W.device
        orig_dtype = W.dtype
        W64 = W.data.to(device=dev, dtype=torch.float64)
        Q64 = Q.to(device=dev, dtype=torch.float64)

        # Slice and rotate: Wi <- Wi @ Q
        h = hidden
        Wq = W64[0:h, :] @ Q64
        Wk = W64[h:2*h, :] @ Q64
        Wv = W64[2*h:3*h, :] @ Q64

        # Stitch back
        W64_rot = torch.empty_like(W64)
        W64_rot[0:h, :] = Wq
        W64_rot[h:2*h, :] = Wk
        W64_rot[2*h:3*h, :] = Wv

        qkv.weight.data = W64_rot.to(device=dev, dtype=orig_dtype)
        # bias unchanged
        return

    # If neither structure is found, do nothing (or raise)
    raise AttributeError("rotate_attention_inputs: could not find q/k/v projections or a packed qkv Linear.")

# def rotate_attention_output(layer, Q, model_type) -> None:
#     # Rotate output matrix of the self-attention layer.
#     if model_type == model_utils.LLAMA_MODEL:
#         W = layer.self_attn.o_proj
#     elif model_type == model_utils.OPT_MODEL:
#         W = layer.self_attn.out_proj
#     else:
#         raise ValueError(f'Unknown model type {model_type}')

#     dtype = W.weight.data.dtype
#     W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
#     W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
#     if W.bias is not None:
#         b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
#         W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    """
    Rotate output matrix of the self-attention layer.
    - LLaMA: layer.self_attn.o_proj
    - OPT:   layer.self_attn.out_proj
    - VGGT:  layer.attn.proj
    """
    import torch

    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    elif model_type == model_utils.VGGT_MODEL:
        # In VGGT Block, the Attention module exposes the output projection as .proj
        attn = getattr(layer, "attn", None)
        if attn is None or not hasattr(attn, "proj"):
            raise AttributeError("Expected layer.attn.proj for VGGT")
        W = attn.proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    # Match original compute path: do math in float64 on current device, then cast back
    dtype = W.weight.data.dtype
    dev = W.weight.data.device
    W_ = W.weight.data.to(device=dev, dtype=torch.float64)
    Q64 = Q.to(device=dev, dtype=torch.float64)

    # Left-multiply by Q^T to rotate output space
    W.weight.data = (Q64.T @ W_).to(device=dev, dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(device=dev, dtype=torch.float64)
        W.bias.data = (Q64.T @ b).to(device=dev, dtype=dtype)

# def rotate_mlp_input(layer, Q, model_type):
#     # Rotate the MLP input weights.
#     if model_type == model_utils.LLAMA_MODEL:
#         mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
#     elif model_type == model_utils.OPT_MODEL:
#         mlp_inputs = [layer.fc1]
#     else:
#         raise ValueError(f'Unknown model type {model_type}')
#     for W in mlp_inputs:
#         dtype = W.weight.dtype
#         W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
#         W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type == model_utils.VGGT_MODEL:
        # In VGGT Block, the MLP uses fc1 as the input projection
        mlp_inputs = [layer.mlp.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')

    for W in mlp_inputs:
        dtype = W.weight.dtype
        dev = W.weight.device
        W_ = W.weight.data.to(device=dev, dtype=torch.float64)
        Q64 = Q.to(device=dev, dtype=torch.float64)
        W.weight.data = (W_ @ Q64).to(device=dev, dtype=dtype)
    
# def rotate_mlp_output(layer, Q, model_type):
#     # Rotate the MLP output weights and bias.
#     if model_type == model_utils.LLAMA_MODEL:
#         W = layer.mlp.down_proj
#     elif model_type == model_utils.OPT_MODEL:
#         W = layer.fc2
#     else:
#         raise ValueError(f'Unknown model type {model_type}')
#     dtype = W.weight.data.dtype
#     W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
#     W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
#     apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
#     if W.bias is not None:
#         b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
#         W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.VGGT_MODEL:
        # In VGGT Block, the MLP output projection is fc2
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    dev = W.weight.data.device

    W_ = W.weight.data.to(device=dev, dtype=torch.float64)
    Q64 = Q.to(device=dev, dtype=torch.float64)

    # Left-multiply by Q^T
    W.weight.data = (Q64.T @ W_).to(device=dev, dtype=dtype)

    # Apply exact (inverse) Hadamard on the weights of MLP output (unchanged helper)
    apply_exact_had_to_linear(W, had_dim=-1, output=False)

    if W.bias is not None:
        b = W.bias.data.to(device=dev, dtype=torch.float64)
        W.bias.data = (Q64.T @ b).to(device=dev, dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 

def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
    
    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

# def rotate_ov_proj(layer, model_type, head_num, head_dim):
#     v_proj = layer.self_attn.v_proj
#     if model_type == model_utils.LLAMA_MODEL:
#         o_proj = layer.self_attn.o_proj
#     elif model_type == model_utils.OPT_MODEL:
#         o_proj = layer.self_attn.out_proj
#     else:
#         raise ValueError(f'Unknown model type {model_type}')
    
#     apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
#     apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def rotate_ov_proj(layer, model_type, head_num, head_dim):
    # For LLaMA/OPT, v_proj is explicit. For VGGT, V is the last third of attn.qkv.
    if model_type == model_utils.LLAMA_MODEL:
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        # Apply Hadamard as before
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    elif model_type == model_utils.OPT_MODEL:
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.out_proj
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    elif model_type == model_utils.VGGT_MODEL:
        # VGGT: v is packed in attn.qkv (Linear(out=3*hidden, in=hidden)), o_proj is attn.proj
        attn = getattr(layer, "attn", None)
        if attn is None or not hasattr(attn, "qkv") or not hasattr(attn, "proj"):
            raise AttributeError("Expected layer.attn.qkv and layer.attn.proj for VGGT")
        qkv = attn.qkv  # Linear
        proj = attn.proj

        # Extract V slice: weights shape [3*H, H], biases [3*H]
        W = qkv.weight
        B = qkv.bias
        out_features, in_features = W.shape
        if out_features % 3 != 0:
            raise ValueError(f"qkv.out_features ({out_features}) not divisible by 3")
        H = in_features
        if out_features != 3 * H:
            raise ValueError(f"Expected qkv.out_features == 3*in_features, got {out_features} vs 3*{H}")

        v_start, v_end = 2 * H, 3 * H

        # Create a temporary Linear that views the V slice so we can reuse apply_exact_had_to_linear
        # Note: We must copy to a temporary module, transform, then write back.
        import torch
        tmp_v = torch.nn.Linear(in_features=H, out_features=H, bias=(B is not None))
        # Initialize with V slice (preserve dtype/device)
        dev = W.device
        dtype = W.dtype
        tmp_v.weight.data = W[v_start:v_end, :].to(device=dev, dtype=dtype).clone()
        if B is not None:
            tmp_v.bias.data = B[v_start:v_end].to(device=dev, dtype=dtype).clone()

        # Apply Hadamard to V projection (output=True matches original intent on v_proj)
        apply_exact_had_to_linear(tmp_v, had_dim=head_dim, output=True)

        # Write back transformed V slice
        W[v_start:v_end, :] = tmp_v.weight.data.to(device=dev, dtype=dtype)
        if B is not None:
            B[v_start:v_end] = tmp_v.bias.data.to(device=dev, dtype=dtype)

        # Apply Hadamard to the output projection (proj) exactly like o_proj before
        apply_exact_had_to_linear(proj, had_dim=-1, output=False)
    else:
        raise ValueError(f'Unknown model type {model_type}')

@torch.inference_mode()
def rotate_model(model, args):
    # Q = get_orthogonal_matrix(model.config.hidden_size,
    #                                             args.rotate_mode)
    # config = model.config
    # num_heads = config.num_attention_heads
    # model_dim = config.hidden_size
    # head_dim = model_dim // num_heads


    # model_type = model_utils.model_type_extractor(model)
    # # rotate_embeddings(model, Q)
    # # rotate_head(model, Q)
    # # utils.cleanup_memory()
    # layers = model_utils.get_transformer_layers(model, 
    #                                             model_type=model_type)
    if hasattr(model, "config"):
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
    else:
        # VGGT path (no model.config): infer from the module structure.
        # Prefer using model_utils helpers if available.
        # 1) Get layers first, so we can inspect a block.
        model_type = model_utils.model_type_extractor(model)
        layers = model_utils.get_transformer_layers(model, model_type=model_type)
        if len(layers) == 0:
            raise ValueError("No transformer layers found to infer sizes.")

        block = layers[0]

        # Infer hidden_size from a canonical Linear in the block (e.g., attn.proj or mlp.fc2).
        hidden_size = None
        num_heads = None

        # Try attention.proj first
        attn = getattr(block, "self_attn", None) or getattr(block, "attn", None)
        if attn is not None:
            proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None) or getattr(attn, "proj", None)
            if proj is not None and isinstance(proj, torch.nn.Linear):
                hidden_size = proj.in_features  # proj: [hidden_size, hidden_size] as Linear(out,in) in PyTorch weight
            # Try to infer num_heads from qkv or q_proj
            qkv = getattr(attn, "qkv", None)
            q_proj = getattr(attn, "q_proj", None)
            if qkv is not None and isinstance(qkv, torch.nn.Linear):
                # qkv.out_features = 3 * hidden_size; typical head_dim = hidden_size // num_heads
                out_f, in_f = qkv.out_features, qkv.in_features
                # Prefer in_f as hidden size if not set by proj
                if hidden_size is None:
                    hidden_size = in_f
                # If divisible, guess num_heads from common head_dim divisors.
                # We can try to find num_heads from attn.num_heads attribute if present.
                num_heads_attr = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
                if isinstance(num_heads_attr, int) and num_heads_attr > 0:
                    num_heads = num_heads_attr
            elif q_proj is not None and isinstance(q_proj, torch.nn.Linear):
                # q_proj.out_features = hidden_size; commonly hidden_size, and head_dim divides it.
                if hidden_size is None:
                    hidden_size = q_proj.out_features
                num_heads_attr = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
                if isinstance(num_heads_attr, int) and num_heads_attr > 0:
                    num_heads = num_heads_attr

        # If still missing hidden_size, fall back to MLP fc2
        if hidden_size is None:
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                fc2 = getattr(mlp, "down_proj", None) or getattr(mlp, "fc2", None)
                if fc2 is not None and isinstance(fc2, torch.nn.Linear):
                    hidden_size = fc2.out_features  # fc2: Linear(out=hidden, in=intermediate)

        if hidden_size is None:
            raise ValueError("Could not infer hidden_size for model without config.")

        # Infer num_heads if still None by probing common attributes on model or block
        if num_heads is None:
            # Try model-level attributes
            num_heads = getattr(model, "num_heads", None) or getattr(model, "num_attention_heads", None)
        if num_heads is None:
            # Heuristic: try to infer from attn.qkv if available using common head_dim set
            if attn is not None:
                qkv = getattr(attn, "qkv", None)
                if qkv is not None and isinstance(qkv, torch.nn.Linear):
                    # Choose num_heads as the largest power-of-two divisor up to 128 that divides hidden_size
                    for nh in (128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2):
                        if hidden_size % nh == 0:
                            num_heads = nh
                            break
        if num_heads is None:
            raise ValueError("Could not infer num_attention_heads for model without config.")

        # Now continue with layers already fetched
        model_dim = hidden_size
        head_dim = model_dim // num_heads

    # Build Q and run rotations
    Q = get_orthogonal_matrix(model_dim, args.rotate_mode)
    
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k



def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)

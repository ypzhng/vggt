import math
import time
import tqdm
import torch
import torch.nn as nn
from . import utils
from . import quant_utils
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _iter_vggt_blocks(model):
    agg = getattr(model, "aggregator", None)
    if agg is None:
        return
    for list_name in ("frame_blocks", "global_blocks"):
        modlist = getattr(agg, list_name, None)
        if isinstance(modlist, nn.ModuleList):
            for idx, block in enumerate(modlist):
                yield f"aggregator.{list_name}.{idx}", block

def _find_linear_modules(module: nn.Module, prefix: str):
    """
    Recursively find all nn.Linear modules under 'module', returning
    a dict: {full_name: linear_module}
    """
    found = {}
    for name, child in module.named_modules():
        # name is relative to module; build full path
        full_name = f"{prefix}.{name}" if name else prefix
        if isinstance(child, nn.Linear):
            found[full_name] = child
    return found

class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.o_proj.module'],
                ['mlp.up_proj.module', 'mlp.gate_proj.module'],
                ['mlp.down_proj.module']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers


def _resolve_attr_path(root, path):
    """
    Resolve dotted attribute/index path like 'aggregator.frame_blocks.3' into a module.
    Supports integer indexing for ModuleList/Sequential.
    """
    obj = root
    for part in path.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(obj, (torch.nn.ModuleList, torch.nn.Sequential)):
                if idx < 0 or idx >= len(obj):
                    return None
                obj = obj[idx]
            else:
                # It might be a regular Python list
                obj = obj[idx]
        else:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
    return obj
       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    """
    RTN weight quantization adapted for VGGT.
    - Only quantizes Linear layers inside aggregator.frame_blocks and aggregator.global_blocks.
    - Uses args.w_bits (or 8 bits for mlp.fc2 if args.int8_down_proj).
    - No groupsize support (must be -1).
    """
    assert getattr(args, "w_groupsize", -1) == -1, "Groupsize not supported in RTN!"
    torch.cuda.empty_cache()

    quantizers = {}

    # Iterate over VGGT aggregator blocks
    blocks = list(_iter_vggt_blocks(model))
    for block_prefix, _ in tqdm.tqdm(blocks, desc="(RtN Quant.) VGGT blocks"):
        # Resolve the block in the model by attribute path and move it IN-PLACE to target device
        block = _resolve_attr_path(model, block_prefix)
        if block is None:
            continue
        # Move the block and its children to the target device
        block.to(dev, non_blocking=True)

        # Collect Linear layers under this block (use the live block reference)
        subset = _find_linear_modules(block, prefix=block_prefix)
        for full_name, lin in subset.items():
            # Decide bits per layer
            layer_weight_bits = args.w_bits
            if getattr(args, "int8_down_proj", False) and full_name.endswith(".mlp.fc2"):
                layer_weight_bits = 8

            if "lm_head" in full_name:
                continue

            # Configure and apply RTN quantizer
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(layer_weight_bits,
                                perchannel=True,
                                sym=not args.w_asym,
                                mse=args.w_clip)

            # Work with the layer's current dtype/device
            W = lin.weight.data
            # Find params on the same device as W
            quantizer.find_params(W)
            # Quantize and cast back to the layer dtype; keep device as-is
            Wq = quantizer.quantize(W).to(dtype=W.dtype, device=W.device)
            lin.weight.data.copy_(Wq)

            # Optional: quantize bias is uncommon; leave as FP
            # Store the quantizer for this layer on CPU to save VRAM
            quantizers[full_name] = quantizer.to("cpu")

        # DO NOT move the block back to CPU here. Keep it on 'dev' for inference.
        # If you must reclaim memory during a very large model quantization, use a two-model approach
        # (CPU scratch copy and copy weights back), but do not leave CPU subtrees in the live model.

        torch.cuda.empty_cache()

    utils.cleanup_memory(verbos=True)
    return quantizers




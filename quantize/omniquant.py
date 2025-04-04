import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer, QuantOPTDecoderLayerPart1, QuantOPTDecoderLayerPart2
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")

from torch.autograd import Variable



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def move_to_device(module, device):
    module.to(device)
    for submodule in module.modules():
        if hasattr(submodule, 'buffer'):
            submodule.buffer = submodule.buffer.to(device)
        if hasattr(submodule, 'parameters'):
            for param in submodule.parameters():
                param.data = param.data.to(device)
    return module

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        """model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)"""
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    #layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0])
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    #layers[0] = layers[0].cpu()
    logger.info("end of input catching!")
    print(f"memory: {torch.cuda.max_memory_allocated(lm._device) / 1024**2}")
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()



    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    print(args)
    
    attention_mask = attention_mask.to("cuda")
    attention_mask_batch = attention_mask_batch.to("cuda")
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)

        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)
        
        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask,position_ids=position_ids)[0]
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            #print(f"{pairs[key]}, {shift}, {scale}")
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            grad_scaler = torch.cuda.amp.GradScaler()

            # Break the layers
            qlayer_parts = [QuantOPTDecoderLayerPart1(qlayer), QuantOPTDecoderLayerPart2(qlayer)]
            """
            optimizers = [torch.optim.AdamW(
                [{"params":let_parameters(qlayer_parts[0], use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer_parts[0]),"lr":args.lwc_lr}],weight_decay=args.wd),
                torch.optim.AdamW(
                [{"params":let_parameters(qlayer_parts[1], use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer_parts[1]),"lr":args.lwc_lr}],weight_decay=args.wd)]
            loss_scalers = [utils.NativeScalerWithGradNormCount(), utils.NativeScalerWithGradNormCount()]
            """

            print(f"max memory: {torch.cuda.max_memory_allocated(lm._device) / 1024**2}")
            print(f"batch size: {args.batch_size}")

            # Put data onto their corresponding devices
            qlayer = qlayer.to("cpu")
            quant_inps = quant_inps.to("cpu")
            fp_inps = fp_inps.to("cpu")
            #print(f"memory after transfering inputs: {torch.cuda.max_memory_allocated(lm._device) / 1024**2}")
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):
                    #print(f"----------sample {j + 1} of {args.nsamples}-------------")
                    index = j * args.batch_size
                    # obtain output of quantization model
                    quant_inp = quant_inps[index:index+args.batch_size,].to("cuda")
                    fp_inp = fp_inps[index:index+args.batch_size,].to("cuda")
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        #print(f"memory after temp quant: {torch.cuda.max_memory_allocated(lm._device) / 1024**2}")
                        
                        # Forward pass
                        # Part 1 forward
                        
                        part1 = qlayer_parts[0].to("cuda")
                        part1_result = part1(quant_inp, attention_mask=attention_mask_batch,position_ids=position_ids)
                        part1 = part1.to("cpu")
                        #print(f"memory after part1: {torch.cuda.memory_allocated(lm._device) / 1024**2}")
                        #print(part1_result)

                        # We need the gradients on the hidden states
                        part1_result[0] = Variable(part1_result[0], requires_grad=True)

                        # Part 2 forward
                        part2 = qlayer_parts[1].to("cuda")
                        part2_result = part2(*part1_result)[0]
                        #print(f"memory after part2: {torch.cuda.memory_allocated(lm._device) / 1024**2}")
                        #print(part2_result)

                        # Loss
                        loss = loss_func(fp_inp, part2_result)
                        #print(f"memory after loss: {torch.cuda.memory_allocated(lm._device) / 1024**2}")
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()         
                    
                    # Backward Pass
                    # Part 2
                    # Unfortunately there doesn't seem to be a good way of doing grad scaling
                    #   across submodels, so we will have to hack it.
                    part2_inputs = [part1_result[0]]
                    #part2_norm = loss_scalers[1](loss, optimizers[1], parameters= get_omni_parameters(part2, use_shift)).to("cpu")
                    scaled_loss = grad_scaler.scale(loss)
                    scaled_loss.backward(create_graph=False, retain_graph=False)
                    #part2 = part2.to("cpu")
                    #print(f"part2 norm {part2_norm}")
                    #print(f"memory after back2: {torch.cuda.memory_allocated(lm._device) / 1024**2}")
                    
                    grad = part2_inputs[0].grad
                    #print(f"grad: {grad}")
                    # Part 1, we first need to do foward once more, since intermediate states are not preserved when the model got swapped between devices
                    with traincast():
                        part1 = part1.to("cuda")
                        part1_result = part1(quant_inp, attention_mask=attention_mask_batch,position_ids=position_ids)
                    part1_output = [part1_result[0]]
                    part1_output[0].backward(gradient = grad, create_graph=False, retain_graph=False)
                    #print(f"memory after back1: {torch.cuda.memory_allocated(lm._device) / 1024**2}")

                    for param_group in optimizer.param_groups:
                        for param in param_group['params']:
                            if param.grad.device != param.device:
                                param.grad = param.grad.to(param.device)

                    # update the grads
                    grad_scaler.unscale_(optimizer)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    # get the norms
                    norm = utils.ampscaler_get_grad_norm(get_omni_parameters(qlayer, use_shift)).cpu()
                    #print(f"norm: {norm}")

                    norm_list.append(norm.data)
                    quant_inp = quant_inp.to("cpu")
                    fp_inp = fp_inp.to("cpu")
                    part1 = part1.to("cpu")
                    part2 = part2.to("cpu")
                    #print(f"max memory so far: {torch.cuda.max_memory_allocated(lm._device) / 1024**2}")

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half()
        part1 = part1.to("cuda")
        part2 = part2.to("cuda")
        qlayer = qlayer.to("cuda")
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0).to("cuda"), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model


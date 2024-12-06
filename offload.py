import torch
from torch import Tensor, nn
from tqdm import tqdm


class PerLayerOffload:
    def __init__(self, model: nn.Module, enable: bool = True) -> None:
        self.enable = enable
        self.model = model
        if not enable:
            return

        # move model to pinned memory. keep a model copy in CPU pinned memory.
        # this can be quite slow if params currently reside in memory-mapped file.
        for p in tqdm(list(model.parameters()), desc="Copying params to pinned memory", dynamic_ncols=True):
            p.data = p.data.cpu().pin_memory()
        self.param_dict = {p: p.data for p in model.parameters()}
        self.manual_params = []

        @torch.compiler.disable()
        def pre_hook(module, args):
            # CPU pinned memory -> CUDA.
            for p in module.parameters():
                p.data = p.data.cuda(non_blocking=True)

        @torch.compiler.disable()
        def post_hook(module, args, output):
            # simply throw CUDA params away.
            # set pointer back to CPU pinned memory copy.
            for p in module.parameters():
                p.data = self.param_dict[p]

        def traverse(module: nn.Module):
            if (
                isinstance(module, (nn.ModuleList, nn.Sequential))
                and len(module) > 1
                and all(type(layer) == type(module[0]) for layer in module)
            ):
                for child in module:
                    child.register_forward_pre_hook(pre_hook)
                    child.register_forward_hook(post_hook)

            else:
                for p in module.parameters(recurse=False):
                    self.manual_params.append(p)
                for child in module.children():
                    traverse(child)

        traverse(model)

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            # CPU pinned memory -> CUDA.
            for p in self.manual_params:
                p.data = p.data.cuda(non_blocking=True)

        return self

    def cpu(self):
        if not self.enable:
            self.model.cpu()

        else:
            # simply throw CUDA params away.
            # set pointer back to CPU pinned memory copy.
            for p in self.manual_params:
                p.data = self.param_dict[p]

        return self


class PerLayerOffloadCUDAStream:
    """This version uses CUDA stream to overlap data transfer with computation.
    Since the hooks are quite different from non-CUDA-stream implementation,
    we use a separate class.
    """

    def __init__(self, model: nn.Module, enable: bool = True, record_stream: bool = False) -> None:
        self.model = model
        self.enable = enable
        if not enable:
            return

        # move model to pinned memory. keep a model copy in CPU pinned memory.
        # this can be quite slow if params currently reside in memory-mapped file.
        for p in tqdm(list(model.parameters()), desc="Copying params to pinned memory", dynamic_ncols=True):
            p.data = p.data.cpu().pin_memory()
        self.param_dict = {p: p.data for p in model.parameters()}
        self.manual_params = []
        self.stream = torch.cuda.Stream()

        def create_pre_hook(next_layer):
            @torch.compiler.disable()
            def pre_hook(module, args):
                # wait for H2D transfer for the current layer to complete
                self.stream.synchronize()

                # start H2D transfer for the next layer
                current_stream = torch.cuda.current_stream()
                with torch.cuda.stream(self.stream):
                    for p in next_layer.parameters():
                        p.data = p.data.cuda(non_blocking=True)

                        # p.data is owned by self.stream
                        # only deallocate once current layer finishes.
                        # compared to torch.cuda.current_stream().synchronize(),
                        # this is slightly faster but uses more memory.
                        if record_stream:
                            p.data.record_stream(current_stream)

            return pre_hook

        @torch.compiler.disable()
        def post_hook(module, args, output):
            if not record_stream:
                torch.cuda.current_stream().synchronize()
            for p in module.parameters():
                p.data = self.param_dict[p]

        def traverse(module: nn.Module):
            if (
                isinstance(module, (nn.ModuleList, nn.Sequential))
                and len(module) > 1
                and all(type(layer) == type(module[0]) for layer in module)
            ):
                # manually move 1st layer params
                self.manual_params.extend(module[0].parameters())

                for i, curr_layer in enumerate(module):
                    # last layer will prefetch 1st layer
                    # -> 1st layer is on GPU before and after sequential
                    next_layer = module[(i + 1) % len(module)]
                    curr_layer.register_forward_pre_hook(create_pre_hook(next_layer))
                    curr_layer.register_forward_hook(post_hook)

            else:
                for p in module.parameters(recurse=False):
                    self.manual_params.append(p)
                for child in module.children():
                    traverse(child)

        traverse(model)

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            for p in self.manual_params:
                p.data = p.data.cuda(non_blocking=True)

        return self

    def cpu(self):
        if not self.enable:
            self.model.cpu()

        else:
            for p in self.manual_params:
                p.data = self.param_dict[p]

        return self


class PerLayerOffloadWithBackward:
    def __init__(self, model: nn.Module, enable: bool = True):
        self.model = model
        self.enable = enable
        if not enable:
            return

        self.stream = torch.cuda.Stream()
        self.manual_params = []
        self.cpu_param_dict = dict()
        self.gpu_param_dict = dict()

        def traverse(module: nn.Module, key: tuple[str, ...] = ()):
            if (
                isinstance(module, (nn.ModuleList, nn.Sequential))
                and len(module) > 1
                and all(type(layer) == type(module[0]) for layer in module)
            ):
                self._register_sequential(module, key)

            else:
                for p in module.parameters(recurse=False):
                    self.manual_params.append(p)
                for name, child in module.named_children():
                    traverse(child, key + (name,))

        traverse(model)

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            for p in self.manual_params:
                p.data = p.data.cuda(non_blocking=True)

            # reset initial state
            for key in self.gpu_param_dict.keys():
                self.gpu_param_dict[key][0].data = self.cpu_param_dict[key][0].cuda(non_blocking=True)
                self.gpu_param_dict[key][1].data = self.cpu_param_dict[key][-1].cuda(non_blocking=True)

        return self

    def cpu(self):
        if not self.enable:
            self.model.cpu()

        else:
            for p in self.manual_params:
                p.data = p.data.cpu()

            for key in self.gpu_param_dict.keys():
                self.gpu_param_dict[key][0].data = self.cpu_param_dict[key][0]
                self.gpu_param_dict[key][1].data = self.cpu_param_dict[key][-1]

        return self

    @staticmethod
    def _get_flat_param(module: nn.Module):
        return torch.cat([x.detach().view(-1) for x in module.parameters()], dim=0)

    @staticmethod
    def _view_into_flat_param(module: nn.Module, flat_param: Tensor):
        offset = 0
        for p in module.parameters():
            p.data = flat_param[offset : offset + p.numel()].view(p.shape)
            offset += p.numel()

    def _register_sequential(self, module: nn.Sequential | nn.ModuleList, key: tuple[str, ...]):
        # double buffering: pre-allocate GPU memory for 2 layers
        # we will alternate between the two: compute on 1st buffer,
        # while transfering 2nd buffer from CPU, then swap.
        self.cpu_param_dict[key] = []
        self.gpu_param_dict[key] = [
            self._get_flat_param(module[0]).cuda(),  # compute buffer
            self._get_flat_param(module[-1]).cuda(),  # transfer buffer
        ]

        def create_pre_forward_hook(idx: int):
            @torch.compiler.disable()
            def pre_forward_hook(module, args):
                # set current layer to the compute buffer
                compute_buffer, transfer_buffer = self.gpu_param_dict[key]
                self._view_into_flat_param(module, compute_buffer)
                current_stream = torch.cuda.current_stream()

                # compute of current layer depends on H2D transfer of current layer
                current_stream.wait_stream(self.stream)

                # H2D transfer of next layer depends on compute of previous layer
                # since they share the same buffer
                self.stream.wait_stream(current_stream)

                with torch.cuda.stream(self.stream):
                    n = len(self.cpu_param_dict[key])
                    cpu_param = self.cpu_param_dict[key][(idx + 1) % n]
                    transfer_buffer.copy_(cpu_param, non_blocking=True)

                # swap compute and transfer buffers
                self.gpu_param_dict[key] = [transfer_buffer, compute_buffer]

            return pre_forward_hook

        def create_pre_backward_hook(idx: int):
            @torch.compiler.disable()
            def pre_backward_hook(module, grad_output):
                # NOTE: the order of compute and transfer buffer is swapped in backward
                transfer_buffer, compute_buffer = self.gpu_param_dict[key]
                self._view_into_flat_param(module, compute_buffer)
                current_stream = torch.cuda.current_stream()

                # synchronize, similar to forward pass
                current_stream.wait_stream(self.stream)
                self.stream.wait_stream(current_stream)

                with torch.cuda.stream(self.stream):
                    n = len(self.cpu_param_dict[key])
                    cpu_param = self.cpu_param_dict[key][(idx - 1) % n]
                    transfer_buffer.copy_(cpu_param, non_blocking=True)

                self.gpu_param_dict[key] = [compute_buffer, transfer_buffer]

            return pre_backward_hook

        for i, curr_layer in enumerate(tqdm(module, desc=f"Copying params to pinned memory {key}", dynamic_ncols=True)):
            flat_param = self._get_flat_param(curr_layer).cpu().pin_memory()
            self.cpu_param_dict[key].append(flat_param)
            curr_layer.register_forward_pre_hook(create_pre_forward_hook(i))
            curr_layer.register_full_backward_pre_hook(create_pre_backward_hook(i))


class PerLayerOffloadWithBackwardGradient:
    def __init__(self, model: nn.Module, enable: bool = True):
        self.model = model
        self.enable = enable
        if not enable:
            return

        self.stream = torch.cuda.Stream()
        self.manual_params = []
        self.cpu_param_dict = dict()
        self.gpu_param_dict = dict()
        self.param2grad = dict()

        def traverse(module: nn.Module, key: tuple[str, ...] = ()):
            if (
                isinstance(module, (nn.ModuleList, nn.Sequential))
                and len(module) > 1
                and all(type(layer) == type(module[0]) for layer in module)
            ):
                self._register_sequential(module, key)

            else:
                for p in module.parameters(recurse=False):
                    self.manual_params.append(p)
                for name, child in module.named_children():
                    traverse(child, key + (name,))

        traverse(model)

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            for p in self.manual_params:
                p.data = p.data.cuda(non_blocking=True)

            # reset initial state
            for key in self.gpu_param_dict.keys():
                self.gpu_param_dict[key][0].data = self.cpu_param_dict[key][0].cuda(non_blocking=True)
                self.gpu_param_dict[key][1].data = self.cpu_param_dict[key][-1].cuda(non_blocking=True)

        return self

    def cpu(self):
        if not self.enable:
            self.model.cpu()

        else:
            for p in self.manual_params:
                p.data = p.data.cpu()

            for key in self.gpu_param_dict.keys():
                self.gpu_param_dict[key][0].data = self.cpu_param_dict[key][0]
                self.gpu_param_dict[key][1].data = self.cpu_param_dict[key][-1]

        return self

    @staticmethod
    def _get_flat_param(module: nn.Module):
        return torch.cat([x.detach().view(-1) for x in module.parameters()], dim=0)

    @staticmethod
    def _view_into_flat_param(module: nn.Module, flat_param: Tensor):
        offset = 0
        for p in module.parameters():
            p.data = flat_param[offset : offset + p.numel()].view(p.shape)
            offset += p.numel()

    def _register_sequential(self, module_list: nn.Sequential | nn.ModuleList, key: tuple[str, ...]):
        # double buffering: pre-allocate GPU memory for 2 layers
        # we will alternate between the two: compute on 1st buffer,
        # while transfering 2nd buffer from CPU, then swap.
        self.cpu_param_dict[key] = []
        self.gpu_param_dict[key] = [
            self._get_flat_param(module_list[0]).cuda(),  # compute buffer
            self._get_flat_param(module_list[-1]).cuda(),  # transfer buffer
        ]

        def create_pre_forward_hook(idx: int):
            @torch.compiler.disable()
            def pre_forward_hook(module, args):
                # set current layer to the compute buffer
                compute_buffer, transfer_buffer = self.gpu_param_dict[key]
                self._view_into_flat_param(module, compute_buffer)

                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(self.stream)
                self.stream.wait_stream(current_stream)

                with torch.cuda.stream(self.stream):
                    n = len(self.cpu_param_dict[key])
                    cpu_param = self.cpu_param_dict[key][(idx + 1) % n]
                    transfer_buffer.copy_(cpu_param, non_blocking=True)

                # swap compute and transfer buffers
                self.gpu_param_dict[key] = [transfer_buffer, compute_buffer]

            return pre_forward_hook

        def create_pre_backward_hook(idx: int):
            @torch.compiler.disable()
            def pre_backward_hook(module, grad_output):
                # NOTE: the order of compute and transfer buffer is swapped in backward
                transfer_buffer, compute_buffer = self.gpu_param_dict[key]
                self._view_into_flat_param(module, compute_buffer)

                current_stream = torch.cuda.current_stream()
                self.stream.wait_stream(current_stream)
                current_stream.wait_stream(self.stream)

                with torch.cuda.stream(self.stream):
                    n = len(module_list)

                    # NOTE: not sure why this won't work in post-backward hook
                    # NOTE: gradients of first layer (last in backward) won't be offloaded
                    # we will perform optim step for first layer on GPU
                    if idx < n - 1:
                        for p in module_list[idx + 1].parameters():
                            if p.grad is None:
                                continue
                            self.param2grad[p].copy_(p.grad, non_blocking=True)

                            # it would be better if we can avoid Tensor.record_stream()
                            p.grad.record_stream(self.stream)
                            p.grad = None

                    cpu_param = self.cpu_param_dict[key][(idx - 1) % n]
                    transfer_buffer.copy_(cpu_param, non_blocking=True)

                self.gpu_param_dict[key] = [compute_buffer, transfer_buffer]

            return pre_backward_hook

        desc = f"Copying params to pinned memory {key}"
        for i, curr_layer in enumerate(tqdm(module_list, desc=desc, dynamic_ncols=True)):
            flat_param = self._get_flat_param(curr_layer).cpu().pin_memory()
            self.cpu_param_dict[key].append(flat_param)

            # pre-allocate pinned memory for gradients, except first layer,
            # because we won't offload gradients of first layer.
            if i > 0:
                for p in curr_layer.parameters():
                    if not p.requires_grad:
                        continue
                    self.param2grad[p] = torch.empty_like(p, device="cpu", pin_memory=True, requires_grad=True)

            curr_layer.register_forward_pre_hook(create_pre_forward_hook(i))
            curr_layer.register_full_backward_pre_hook(create_pre_backward_hook(i))

        # TODO: register hooks for optimizer to view CPU tensor


if __name__ == "__main__":
    from flux_infer import flux_img_ids
    from modelling import Flux, FluxConfig

    # also need to set CUBLAS_WORKSPACE_CONFIG=:4096:8
    torch.use_deterministic_algorithms(True)

    config = FluxConfig(depth=3, depth_single_blocks=6, hidden_size=2048, num_heads=16)
    flux = Flux(config).bfloat16().cuda()

    height = width = 1024
    latents = torch.randn(1, height // 16 * width // 16, 64).bfloat16().cuda()
    img_ids = flux_img_ids(1, height // 16, width // 16).bfloat16().cuda()
    txt = torch.randn(1, 512, 4096).bfloat16().cuda()
    txt_ids = torch.zeros(1, txt.shape[1], 3).bfloat16().cuda()
    vec = torch.randn(1, 768).bfloat16().cuda()
    t_vec = torch.rand(1).bfloat16().cuda()
    guidance = (torch.ones(1) * 3.5).bfloat16().cuda()

    out = flux(latents, img_ids, txt, txt_ids, t_vec, vec, guidance)
    out.sum().backward()
    grad = flux.img_in.weight.grad

    for p in flux.parameters():
        p.grad = None
    flux.cpu()
    # PerLayerOffloadWithBackward(flux).cuda()
    PerLayerOffloadWithBackwardGradient(flux).cuda()

    out2 = flux(latents, img_ids, txt, txt_ids, t_vec, vec, guidance)
    out2.sum().backward()
    grad2 = flux.img_in.weight.grad

    torch.testing.assert_close(out, out2)
    torch.testing.assert_close(grad, grad2)

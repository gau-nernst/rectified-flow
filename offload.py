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

        manual_params = set()  # we must use a set here in case of tied parameters

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
                    manual_params.add(p)
                for child in module.children():
                    traverse(child)

        traverse(model)
        self.manual_tensors = list(manual_params) + list(self.model.buffers())

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            # CPU pinned memory -> CUDA.
            for p in self.manual_tensors:
                p.data = p.data.cuda(non_blocking=True)

        return self

    def cpu(self):
        if not self.enable:
            self.model.cpu()

        else:
            # simply throw CUDA params away.
            # set pointer back to CPU pinned memory copy.
            for p in self.manual_tensors:
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
        self.stream = torch.cuda.Stream()

        def create_pre_hook(next_layer):
            @torch.compiler.disable()
            def pre_hook(module, args):
                # wait for H2D transfer for the current layer to complete
                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(self.stream)

                # start H2D transfer for the next layer
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

        manual_params = set()

        def traverse(module: nn.Module):
            if (
                isinstance(module, (nn.ModuleList, nn.Sequential))
                and len(module) > 1
                and all(type(layer) == type(module[0]) for layer in module)
            ):
                # manually move 1st layer params
                manual_params.update(module[0].parameters())

                for i, curr_layer in enumerate(module):
                    # last layer will prefetch 1st layer
                    # -> 1st layer is on GPU before and after sequential
                    next_layer = module[(i + 1) % len(module)]
                    curr_layer.register_forward_pre_hook(create_pre_hook(next_layer))
                    curr_layer.register_forward_hook(post_hook)

            else:
                for p in module.parameters(recurse=False):
                    manual_params.update(p)
                for child in module.children():
                    traverse(child)

        traverse(model)
        self.manual_tensors = list(manual_params) + list(self.model.buffers())

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            for p in self.manual_tensors:
                p.data = p.data.cuda(non_blocking=True)

        return self

    def cpu(self):
        if not self.enable:
            self.model.cpu()

        else:
            for p in self.manual_tensors:
                p.data = self.param_dict[p]

        return self


class PerLayerOffloadWithBackward:
    """This version adds supports for backward pass, which will prefetch sequential layers in reverse order."""

    def __init__(self, model: nn.Module, enable: bool = True):
        self.model = model
        self.enable = enable
        if not enable:
            return

        self.stream = torch.cuda.Stream()
        self.cpu_param_dict = dict()
        self.gpu_param_dict = dict()

        manual_params = set()

        def traverse(module: nn.Module, key: tuple[str, ...] = ()):
            if (
                isinstance(module, (nn.ModuleList, nn.Sequential))
                and len(module) > 1
                and all(type(layer) == type(module[0]) for layer in module)
            ):
                self._register_sequential(module, key)

            else:
                for p in module.parameters(recurse=False):
                    manual_params.add(p)
                for name, child in module.named_children():
                    traverse(child, key + (name,))

        traverse(model)
        self.manual_tensors = list(manual_params) + list(self.model.buffers())

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            for p in self.manual_tensors:
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
            for p in self.manual_tensors:
                p.data = p.data.cpu()

            for key in self.gpu_param_dict.keys():
                self.gpu_param_dict[key][0].data = self.cpu_param_dict[key][0]
                self.gpu_param_dict[key][1].data = self.cpu_param_dict[key][-1]

        return self

    @staticmethod
    def _get_flat_param(module: nn.Module):
        return torch.cat([x.detach().view(-1) for x in module.parameters()], dim=0)

    @staticmethod
    @torch.compiler.disable()
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
    "This version also offloads gradients. To ensure proper synchronization, it will take control over the optimizer."

    def __init__(
        self,
        model: nn.Module,
        optim_cls: type[torch.optim.Optimizer],
        optim_kwargs: dict | None = None,
        enable: bool = True,
    ):
        self.model = model
        self.enable = enable
        if not enable:
            return

        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs

        self.stream = torch.cuda.Stream()
        self.disable_forward_hook = False

        self.key2flat_gpu_buffer = dict()
        self.key2flat_cpu_params = dict()

        self.param2cpu_view = dict()
        self.param2optim = dict()
        self.param_queue = []  # we will run optimizer in this order

        manual_params = set()

        def traverse(module: nn.Module, key: tuple[str, ...] = ()):
            if (
                isinstance(module, (nn.ModuleList, nn.Sequential))
                and len(module) > 1
                and all(type(layer) == type(module[0]) for layer in module)
            ):
                self._register_sequential(module, key)

            else:
                for p in module.parameters(recurse=False):
                    manual_params.add(p)
                for name, child in module.named_children():
                    traverse(child, key + (name,))

        traverse(model)
        self.manual_tensors = list(manual_params) + list(self.model.buffers())
        self.manual_optim = optim_cls(manual_params, **(optim_kwargs or dict()))

    def cuda(self):
        if not self.enable:
            self.model.cuda()

        else:
            for p in self.manual_tensors:
                p.data = p.data.cuda(non_blocking=True)

        return self

    def cpu(self):
        if not self.enable:
            self.model.cpu()

        else:
            for p in self.manual_tensors:
                p.data = self.param2cpu.get(p, p.data.cpu())

        return self

    @staticmethod
    def _get_flat_param(module: nn.Module):
        return torch.cat([x.detach().view(-1) for x in module.parameters()], dim=0)

    @staticmethod
    @torch.compiler.disable()
    def _view_into_flat_param(module: nn.Module, flat_param: Tensor):
        offset = 0
        for p in module.parameters():
            p.data = flat_param[offset : offset + p.numel()].view(p.shape)
            offset += p.numel()

    def _register_sequential(self, module_list: nn.Sequential | nn.ModuleList, key: tuple[str, ...]):
        self.key2flat_gpu_buffer[key] = [
            self._get_flat_param(module_list[0]).cuda(),
            self._get_flat_param(module_list[-1]).cuda(),
        ]
        self.key2flat_cpu_params[key] = []

        def create_pre_forward_hook(idx: int):
            def pre_forward_hook(module: nn.Module, inputs: tuple):
                # when there is activation checkpointing, .forward() is re-run in backward pass.
                # we use this flag to disable forward hooks in this case. set it to True before
                # calling loss.backward() and set it back to False after that.
                if self.disable_forward_hook:
                    return

                compute_buffer, transfer_buffer = self.key2flat_gpu_buffer[key]
                self._view_into_flat_param(module, compute_buffer)

                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(self.stream)
                self.stream.wait_stream(current_stream)

                with torch.cuda.stream(self.stream):
                    next_layer_cpu = self.key2flat_cpu_params[key][(idx + 1) % len(module_list)]
                    transfer_buffer.copy_(next_layer_cpu, non_blocking=True)

                self.key2flat_gpu_buffer[key] = [transfer_buffer, compute_buffer]

            return pre_forward_hook

        def create_pre_backward_hook(idx: int):
            def pre_backward_hook(module, grad_output):
                transfer_buffer, compute_buffer = self.key2flat_gpu_buffer[key]
                self._view_into_flat_param(module, compute_buffer)

                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(self.stream)
                self.stream.wait_stream(current_stream)

                with torch.cuda.stream(self.stream):
                    next_layer_cpu = self.key2flat_cpu_params[key][(idx - 1) % len(module_list)]
                    transfer_buffer.copy_(next_layer_cpu, non_blocking=True)

                self.key2flat_gpu_buffer[key] = [compute_buffer, transfer_buffer]

            return pre_backward_hook

        # NOTE: apparently when nn.Module.register_full_backward_hook() fires, param.grad
        # is not guaranteed to be computed https://github.com/pytorch/pytorch/issues/86051
        # hence, we have to use Tensor.register_post_accumulate_grad_hook() to offload grads.
        def post_grad_hook(p: Tensor):
            # make sure p.grad finished being computed
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.param2cpu_view[p].grad.copy_(p.grad, non_blocking=True)

            # we will execute optim step in this order
            self.param_queue.append((p, self.stream.record_event()))

            # free grad memory
            p.grad.record_stream(self.stream)
            p.grad = None

            # if we want to do optimizer step on GPU
            # we must fuse optimizer with backward (params still on CUDA)
            # we can't prefetch the next layer until optim-step for previous layer finishes
            # optim-step must wait for all params have their grads ready
            # we can use post-grad-hook to count the number of params NOT having grads ready left
            # when the number reaches 0 -> we can start optimizer step
            # after optimizer step, we can record an event.
            # however, it's hard to synchronize this event with prefecthing the next layer
            # we probably have to give up fixed buffer

        desc = f"Copying params to pinned memory {key}"
        for i, curr_layer in enumerate(tqdm(module_list, desc=desc, dynamic_ncols=True)):
            flat_param = self._get_flat_param(curr_layer).cpu().pin_memory()
            self.key2flat_cpu_params[key].append(flat_param)

            offset = 0
            for p in curr_layer.parameters():
                cpu_param = flat_param[offset : offset + p.numel()].view(p.shape)
                offset += p.numel()
                self.param2cpu_view[p] = cpu_param

                # pre-allocate pinned memory for gradients, and install hooks to offload grads
                if p.requires_grad:
                    cpu_param.grad = torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=True)
                    self.param2optim[p] = self.optim_cls([cpu_param], **(self.optim_kwargs or dict()))
                    p.register_post_accumulate_grad_hook(post_grad_hook)

            curr_layer.register_forward_pre_hook(create_pre_forward_hook(i))
            curr_layer.register_full_backward_pre_hook(create_pre_backward_hook(i))

    @torch.no_grad()
    def optim_step(self):
        after_bwd_event = torch.cuda.current_stream().record_event()
        self.manual_optim.step()

        for p, sync_event in self.param_queue:
            sync_event.synchronize()  # wait for grad offload to finish
            self.param2optim[p].step()

        # manually prefetch 1st layer, since it won't be prefetched in pre-forward hook
        # make sure backward finishes
        self.stream.wait_event(after_bwd_event)
        with torch.cuda.stream(self.stream):
            for key in self.key2flat_cpu_params.keys():
                self.key2flat_gpu_buffer[key][0].copy_(self.key2flat_cpu_params[key][0], non_blocking=True)

    def optim_zero_grad(self):
        self.manual_optim.zero_grad()
        self.param_queue = []

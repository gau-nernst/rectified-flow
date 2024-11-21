import torch
from torch import nn
from tqdm import tqdm


class PerLayerOffload:
    def __init__(self, model: nn.Module, enable: bool = True) -> None:
        self.enable = enable
        self.model = model
        if not enable:
            return

        # move model to pinned memory. keep a model copy in CPU pinned memory.
        # this can be quite slow if params currently reside in memory-mapped file.
        for p in tqdm(list(model.parameters()), desc="Copying params to pinned memory"):
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
            if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 1:
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
        for p in tqdm(list(model.parameters()), desc="Copying params to pinned memory"):
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
            if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 1:
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

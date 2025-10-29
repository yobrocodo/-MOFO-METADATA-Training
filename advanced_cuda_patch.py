import sys
import types
import importlib

def patch_cuda():
    import torch
    import torch.nn as nn

    # Backup original methods BEFORE patching
    original_is_available = torch.cuda.is_available
    original_to = torch.Tensor.to
    original_device = torch.device
    original_device_type = torch.device  # <-- This line is added
    original_cuda = nn.Module.cuda

    # Create patched versions
    def patched_is_available():
        return True

    def dummy_init():
        pass

    def patched_device(device_type, index=None, *args, **kwargs):
        if device_type == 'cuda' or str(device_type).startswith('cuda:'):
            return original_device('cpu')
        return original_device(device_type, index, *args, **kwargs)

    def patched_to(self, device=None, *args, **kwargs):
        # Use original_device_type in isinstance
        if device == 'cuda' or (isinstance(device, original_device_type) and device.type == 'cuda'):
            device = 'cpu'
        elif isinstance(device, str) and device.startswith('cuda:'):
            device = 'cpu'
        return original_to(self, device, *args, **kwargs)

    def patched_cuda(self, device=None):
        return self.cpu()

    # Apply patches
    torch.cuda.is_available = patched_is_available
    torch.cuda._lazy_init = dummy_init
    torch.device = patched_device
    torch.Tensor.to = patched_to  # Safely patch only the method, not the class
    nn.Module.cuda = patched_cuda

    # Setup dummy cudnn backend
    if not hasattr(torch, 'backends'):
        torch.backends = types.ModuleType('backends')
    if not hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn = types.ModuleType('cudnn')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.version = lambda: 8000  # Dummy version

    # Additional dummy CUDA functions
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_name = lambda device=None: "CPU (CUDA emulated)"
    torch.cuda.device = lambda device=None: None  # Do nothing
    torch.cuda.set_device = lambda device: None   # Do nothing

    print("Advanced CUDA patching complete")

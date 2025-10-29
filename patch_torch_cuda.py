import sys
import types

# First attempt to patch torch.cuda
try:
    import torch
    
    # Create a backup of the original is_available function
    original_is_available = torch.cuda.is_available
    
    # Create a patched version that always returns True
    def patched_is_available():
        return True
    
    # Apply the patch
    torch.cuda.is_available = patched_is_available
    
    # Add a dummy _lazy_init function if it doesn't exist
    if not hasattr(torch.cuda, '_lazy_init'):
        torch.cuda._lazy_init = lambda: None
    
    # Create a dummy current_device function if needed
    if not hasattr(torch.cuda, 'current_device'):
        torch.cuda.current_device = lambda: 0
    
    # Create a dummy device_count function if needed
    if not hasattr(torch.cuda, 'device_count'):
        torch.cuda.device_count = lambda: 1
    
    print("Successfully patched torch.cuda.is_available to return True")
    
    # Patch torch.backends.cudnn as well
    if not hasattr(torch, 'backends'):
        torch.backends = types.ModuleType('backends')
    if not hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn = types.ModuleType('cudnn')
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    print("Successfully patched torch.backends.cudnn")
    
except Exception as e:
    print(f"Error patching torch.cuda: {str(e)}")
    sys.exit(1)

print("Torch CUDA patching complete - you can now run your script")
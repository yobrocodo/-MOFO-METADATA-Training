import sys
import runpy
import importlib.util

# First, apply our advanced CUDA patches
spec = importlib.util.spec_from_file_location("advanced_cuda_patch", "advanced_cuda_patch.py")
patch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_module)

# ✅ Call the patch function to actually apply it
patch_module.patch_cuda()

# Now run the specified script
if len(sys.argv) > 1:
    script_name = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove our script name from argv

    try:
        print(f"Running {script_name}...")
        runpy.run_path(script_name, run_name="__main__")
    except Exception as e:
        print(f"Error running {script_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("Usage: python run_with_cuda_patch.py <script_to_run.py> [args...]")
    sys.exit(1)

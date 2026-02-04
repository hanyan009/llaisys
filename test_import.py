import sys
print("Python version:", sys.version)
print("Step 1: Import llaisys...")

try:
    import llaisys
    print("  llaisys imported successfully")
    print("  Available attributes:", dir(llaisys))
except Exception as e:
    print(f"  Error importing llaisys: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 2: Import llaisys.models...")
try:
    import llaisys.models
    print("  llaisys.models imported successfully")
    print("  Available models:", dir(llaisys.models))
except Exception as e:
    print(f"  Error importing llaisys.models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Import Qwen2 class...")
try:
    from llaisys.models import Qwen2
    print("  Qwen2 class imported successfully")
except Exception as e:
    print(f"  Error importing Qwen2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful!")

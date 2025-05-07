"""
Test script to verify that all necessary modules can be imported correctly
"""

# Try to import all the required packages
print("Testing imports...")

try:
    import torch
    print(f"✅ torch {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")

try:
    import transformers
    print(f"✅ transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ transformers: {e}")

try:
    import datasets
    print(f"✅ datasets {datasets.__version__}")
except ImportError as e:
    print(f"❌ datasets: {e}")

try:
    import peft
    print(f"✅ peft {peft.__version__}")
except ImportError as e:
    print(f"❌ peft: {e}")

try:
    import trl
    print(f"✅ trl {trl.__version__}")
except ImportError as e:
    print(f"❌ trl: {e}")

try:
    import bitsandbytes
    print(f"✅ bitsandbytes {bitsandbytes.__version__}")
except ImportError as e:
    print(f"❌ bitsandbytes: {e}")

try:
    import fastapi
    print(f"✅ fastapi {fastapi.__version__}")
except ImportError as e:
    print(f"❌ fastapi: {e}")

# Try to import our local modules
print("\nTesting local modules...")

try:
    from model.training import train
    print(f"✅ model.training.train")
except ImportError as e:
    print(f"❌ model.training.train: {e}")

try:
    from utils import preprocessing
    print(f"✅ utils.preprocessing")
except ImportError as e:
    print(f"❌ utils.preprocessing: {e}")

print("\nImport test complete.") 
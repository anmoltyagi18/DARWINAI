import sys
import os

try:
    import ML
    print("SUCCESS: ML imported")
except ImportError as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

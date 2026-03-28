import shutil
import os

src = 'backtrader'
dst = '_backtrader_source'

if not os.path.exists(dst):
    shutil.copytree(src, dst)
    print("Copied successfully.")
else:
    print("Destination already exists.")

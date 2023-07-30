import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_file_name(path: str) -> str:
    return os.path.basename(path).split('.')[0]

import torch
from pathlib import Path
import os

# Get project root (alternative method)
PROJECT_ROOT = Path(os.getcwd()).resolve()  # Use current working directory as the root

# Alternatively, if you know the project root relative to the script:
# PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Use this if running from a script

CONFIG = {
    'input_folder': PROJECT_ROOT / 'alpaca-dataset-generator' / 'data' / 'input',
    'output_file': PROJECT_ROOT / 'alpaca-dataset-generator' /'output' / 'raw_dataset.jsonl',
    'validated_output_file': PROJECT_ROOT / 'output' / 'validated_dataset.jsonl',
    'num_examples': 10000,
    'batch_size': 32,
    'max_workers': 4,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'models': {
        'gpt2': 'gpt2-xl',
        't5': 't5-large',
        'sentiment': 'distilbert/distilbert-base-multilingual-cased',
        'sentence': 'all-mpnet-base-v2'
    }
}

# Create directories if they don't exist
CONFIG['input_folder'].mkdir(parents=True, exist_ok=True)
(CONFIG['output_file'].parent).mkdir(parents=True, exist_ok=True)

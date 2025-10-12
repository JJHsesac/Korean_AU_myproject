import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_model_config(config, model_name):
    """Get specific model configuration"""
    return config['model'][model_name]

def get_training_config(config):
    """Get training configuration"""
    return config['training']

# 사용 예시
if __name__ == "__main__":
    config = load_config()
    print("KcBERT path:", config['model']['kcbert']['path'])
    print("Batch size:", config['training']['batch_size'])
    print("Ensemble weights:", config['model']['ensemble']['weights'])

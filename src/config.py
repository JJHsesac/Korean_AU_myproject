import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    """실험 설정"""
    name: str
    model_name: str
    dataset_dir: str
    learning_rate: float
    batch_size: int
    epochs: int
    max_len: int = 512
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    seed: int = 42
    description: str = ""
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """YAML 파일에서 설정 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 공통 설정
        common = {
            'model_name': config['model_name'],
            'dataset_dir': config['dataset_dir'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'max_len': config.get('max_len', 512),
            'warmup_steps': config.get('warmup_steps', 500),
            'weight_decay': config.get('weight_decay', 0.01),
            'fp16': config.get('fp16', True),
            'seed': config.get('seed', 42),
        }
        
        # 실험별 설정
        experiments = []
        for exp in config['experiments']:
            exp_config = {**common, **exp}
            experiments.append(cls(**exp_config))
        
        return experiments

def load_config(yaml_path: str):
    """편의 함수"""
    return ExperimentConfig.from_yaml(yaml_path)

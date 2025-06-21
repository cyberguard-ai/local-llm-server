import os
import yaml
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    path: str
    context_length: int
    threads: int

@dataclass
class InferenceConfig:
    temperature: float
    top_p: float
    max_tokens: int

@dataclass
class Config:
    model: ModelConfig
    inference: InferenceConfig

def load_config(config_path: str = "config/model_config.yaml") -> Config:
    """Load configuration from YAML file"""
    
    # Get number of CPU threads
    cpu_count = os.cpu_count() or 4
    
    # Default configuration
    default_config = {
        "model": {
            "name": "open-llama-3b-v2",
            "path": os.getenv("MODEL_PATH", os.path.join("models", "model.gguf")),
            "context_length": int(os.getenv("MAX_CONTEXT", "2048")),
            "threads": int(os.getenv("N_THREADS", str(cpu_count)))
        },
        "inference": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 256
        }
    }
    
    # Try to load from file, fall back to defaults
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in file_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                        
                # Handle "auto" threads setting
                if default_config["model"]["threads"] == "auto":
                    default_config["model"]["threads"] = cpu_count
                elif isinstance(default_config["model"]["threads"], str):
                    default_config["model"]["threads"] = int(default_config["model"]["threads"])
                    
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        print("Using default configuration")
    
    return Config(
        model=ModelConfig(**default_config["model"]),
        inference=InferenceConfig(**default_config["inference"])
    )
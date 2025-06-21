from src.utils.config import load_config

config = load_config()
print(f"Model: {config.model.name}")
print(f"Path: {config.model.path}")
print(f"Context Length: {config.model.context_length}")
print(f"Threads: {config.model.threads}")
print(f"Temperature: {config.inference.temperature}")
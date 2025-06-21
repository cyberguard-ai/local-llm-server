"""Unit tests for configuration management."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from src.utils.config import (
    ModelConfig, 
    InferenceConfig, 
    ServerConfig, 
    Config, 
    ConfigLoader,
    load_config
)


class TestModelConfig:
    """Test ModelConfig validation."""
    
    def test_valid_model_config(self):
        """Test creating a valid model configuration."""
        config = ModelConfig(
            name="test-model",
            path="/path/to/model.gguf",
            context_length=2048,
            threads=4
        )
        assert config.name == "test-model"
        assert config.path == "/path/to/model.gguf"
        assert config.context_length == 2048
        assert config.threads == 4
    
    def test_invalid_context_length(self):
        """Test validation fails for invalid context_length."""
        with pytest.raises(ValueError, match="context_length must be positive"):
            ModelConfig(
                name="test-model",
                path="/path/to/model.gguf",
                context_length=0,
                threads=4
            )
        
        with pytest.raises(ValueError, match="context_length must be positive"):
            ModelConfig(
                name="test-model",
                path="/path/to/model.gguf",
                context_length=-100,
                threads=4
            )
    
    def test_invalid_threads(self):
        """Test validation fails for invalid threads."""
        with pytest.raises(ValueError, match="threads must be positive"):
            ModelConfig(
                name="test-model",
                path="/path/to/model.gguf",
                context_length=2048,
                threads=0
            )
        
        with pytest.raises(ValueError, match="threads must be positive"):
            ModelConfig(
                name="test-model",
                path="/path/to/model.gguf",
                context_length=2048,
                threads=-1
            )
    
    def test_empty_path(self):
        """Test validation fails for empty path."""
        with pytest.raises(ValueError, match="model path cannot be empty"):
            ModelConfig(
                name="test-model",
                path="",
                context_length=2048,
                threads=4
            )


class TestInferenceConfig:
    """Test InferenceConfig validation."""
    
    def test_valid_inference_config(self):
        """Test creating a valid inference configuration."""
        config = InferenceConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 256
    
    def test_default_values(self):
        """Test default values are set correctly."""
        config = InferenceConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 256
    
    def test_invalid_temperature(self):
        """Test validation fails for invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            InferenceConfig(temperature=-0.1)
        
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            InferenceConfig(temperature=2.1)
    
    def test_invalid_top_p(self):
        """Test validation fails for invalid top_p."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            InferenceConfig(top_p=-0.1)
        
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            InferenceConfig(top_p=1.1)
    
    def test_invalid_max_tokens(self):
        """Test validation fails for invalid max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            InferenceConfig(max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            InferenceConfig(max_tokens=-10)


class TestServerConfig:
    """Test ServerConfig validation."""
    
    def test_valid_server_config(self):
        """Test creating a valid server configuration."""
        config = ServerConfig(
            host="127.0.0.1",
            port=8080,
            log_level="DEBUG"
        )
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.log_level == "DEBUG"
    
    def test_default_values(self):
        """Test default values are set correctly."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "INFO"
    
    def test_invalid_port(self):
        """Test validation fails for invalid port."""
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ServerConfig(port=0)
        
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ServerConfig(port=65536)
        
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ServerConfig(port=-1)
    
    def test_invalid_log_level(self):
        """Test validation fails for invalid log_level."""
        with pytest.raises(ValueError, match="invalid log_level"):
            ServerConfig(log_level="INVALID")


class TestConfigLoader:
    """Test ConfigLoader functionality."""
    
    @patch('os.cpu_count')
    def test_get_cpu_count(self, mock_cpu_count):
        """Test CPU count calculation."""
        # Test normal case
        mock_cpu_count.return_value = 8
        assert ConfigLoader._get_cpu_count() == 6  # 75% of 8
        
        # Test single core
        mock_cpu_count.return_value = 1
        assert ConfigLoader._get_cpu_count() == 1  # min 1
        
        # Test high core count (should cap at 16)
        mock_cpu_count.return_value = 32
        assert ConfigLoader._get_cpu_count() == 16  # max 16
        
        # Test None case
        mock_cpu_count.return_value = None
        assert ConfigLoader._get_cpu_count() == 3  # 75% of fallback 4
    
    def test_normalize_threads(self):
        """Test thread normalization."""
        with patch.object(ConfigLoader, '_get_cpu_count', return_value=4):
            # Test auto detection
            assert ConfigLoader._normalize_threads("auto") == 4
            assert ConfigLoader._normalize_threads("automatic") == 4
            assert ConfigLoader._normalize_threads("AUTO") == 4
            
            # Test string numbers
            assert ConfigLoader._normalize_threads("8") == 8
            assert ConfigLoader._normalize_threads("1") == 1
            
            # Test integers
            assert ConfigLoader._normalize_threads(4) == 4
            assert ConfigLoader._normalize_threads(12) == 12
            
            # Test invalid strings
            assert ConfigLoader._normalize_threads("invalid") == 4
            assert ConfigLoader._normalize_threads("") == 4
    
    def test_preprocess_config(self):
        """Test configuration preprocessing."""
        config_data = {
            "model": {
                "threads": "auto",
                "context_length": "2048"
            },
            "inference": {
                "temperature": "0.8",
                "top_p": "0.95",
                "max_tokens": "512"
            },
            "server": {
                "port": "9000"
            }
        }
        
        with patch.object(ConfigLoader, '_get_cpu_count', return_value=8):
            processed = ConfigLoader._preprocess_config(config_data)
        
        assert processed["model"]["threads"] == 8
        assert processed["model"]["context_length"] == 2048
        assert processed["inference"]["temperature"] == 0.8
        assert processed["inference"]["top_p"] == 0.95
        assert processed["inference"]["max_tokens"] == 512
        assert processed["server"]["port"] == 9000
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base = {
            "model": {
                "name": "base-model",
                "threads": 4
            },
            "inference": {
                "temperature": 0.7
            }
        }
        
        override = {
            "model": {
                "threads": 8
            },
            "server": {
                "port": 9000
            }
        }
        
        merged = ConfigLoader._merge_configs(base, override)
        
        assert merged["model"]["name"] == "base-model"  # kept from base
        assert merged["model"]["threads"] == 8  # overridden
        assert merged["inference"]["temperature"] == 0.7  # kept from base
        assert merged["server"]["port"] == 9000  # added from override
    
    def test_load_yaml_config_success(self):
        """Test successful YAML loading."""
        yaml_content = """
        model:
          name: "test-model"
          threads: 4
        inference:
          temperature: 0.8
        """
        
        # Use a temp directory instead of NamedTemporaryFile for Windows compatibility
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text(yaml_content, encoding='utf-8')
            
            config = ConfigLoader._load_yaml_config(str(config_path))
            assert config is not None
            assert config["model"]["name"] == "test-model"
            assert config["model"]["threads"] == 4
            assert config["inference"]["temperature"] == 0.8
    
    def test_load_yaml_config_file_not_found(self):
        """Test YAML loading with non-existent file."""
        config = ConfigLoader._load_yaml_config("/nonexistent/file.yaml")
        assert config is None
    
    def test_load_yaml_config_invalid_yaml(self):
        """Test YAML loading with invalid YAML."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid_config.yaml"
            config_path.write_text(invalid_yaml, encoding='utf-8')
            
            config = ConfigLoader._load_yaml_config(str(config_path))
            assert config is None
    
    def test_environment_variable_override(self):
        """Test environment variable configuration."""
        # Clear existing env vars that might interfere
        env_patches = {
            'MODEL_NAME': 'env-model',
            'SERVER_PORT': '9000',
            'DEFAULT_TEMPERATURE': '0.8',
            'MODEL_PATH': 'env/path/model.gguf'  # Add this to ensure it's not picked up from actual config
        }
        
        # Patch environment and ensure we're not loading existing config files
        with patch.dict(os.environ, env_patches, clear=False):
            with patch.object(ConfigLoader, 'DEFAULT_CONFIG_PATHS', []):  # Don't load any config files
                config = ConfigLoader.load()
                
                assert config.model.name == "env-model"
                assert config.server.port == 9000
                assert config.inference.temperature == 0.8
    
    def test_load_with_yaml_file(self):
        """Test loading configuration with YAML file."""
        yaml_content = """
        model:
          name: "yaml-model"
          path: "yaml/path/model.gguf"
          context_length: 1024
          threads: auto
        inference:
          temperature: 0.6
          max_tokens: 128
        server:
          port: 7000
          log_level: "DEBUG"
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text(yaml_content, encoding='utf-8')
            
            config = ConfigLoader.load(str(config_path))
            
            assert config.model.name == "yaml-model"
            assert "yaml" in config.model.path and "model.gguf" in config.model.path
            assert config.model.context_length == 1024
            assert config.model.threads > 0  # auto-detected
            
            assert config.inference.temperature == 0.6
            assert config.inference.max_tokens == 128
            
            assert config.server.port == 7000
            assert config.server.log_level == "DEBUG"
    
    def test_load_config_convenience_function(self):
        """Test the convenience load_config function."""
        config = load_config()
        assert isinstance(config, Config)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.inference, InferenceConfig)
        assert isinstance(config.server, ServerConfig)


class TestConfig:
    """Test complete Config object."""
    
    def test_complete_config_creation(self):
        """Test creating a complete configuration."""
        model_config = ModelConfig(
            name="test-model",
            path="/path/to/model.gguf",
            context_length=2048,
            threads=4
        )
        
        inference_config = InferenceConfig(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512
        )
        
        server_config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            log_level="DEBUG"
        )
        
        config = Config(
            model=model_config,
            inference=inference_config,
            server=server_config
        )
        
        assert config.model == model_config
        assert config.inference == inference_config
        assert config.server == server_config
    
    def test_config_with_default_server(self):
        """Test Config with default ServerConfig."""
        model_config = ModelConfig(
            name="test-model",
            path="/path/to/model.gguf",
            context_length=2048,
            threads=4
        )
        
        inference_config = InferenceConfig()
        
        config = Config(
            model=model_config,
            inference=inference_config
        )
        
        assert isinstance(config.server, ServerConfig)
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000
        assert config.server.log_level == "INFO"


# Integration tests
class TestConfigIntegration:
    """Integration tests for the complete configuration system."""
    
    def test_end_to_end_config_loading(self):
        """Test complete configuration loading workflow."""
        yaml_content = """
        model:
          name: "integration-test-model"
          path: "models/test-model.gguf"
          context_length: 4096
          threads: auto
        
        inference:
          temperature: 0.75
          top_p: 0.85
          max_tokens: 1024
        
        server:
          host: "localhost"
          port: 8080
          log_level: "WARNING"
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "integration_config.yaml"
            config_path.write_text(yaml_content, encoding='utf-8')
            
            # Load configuration
            config = load_config(str(config_path))
            
            # Verify all components
            assert config.model.name == "integration-test-model"
            assert "models" in config.model.path and "test-model.gguf" in config.model.path
            assert config.model.context_length == 4096
            assert config.model.threads > 0
            
            assert config.inference.temperature == 0.75
            assert config.inference.top_p == 0.85
            assert config.inference.max_tokens == 1024
            
            assert config.server.host == "localhost"
            assert config.server.port == 8080
            assert config.server.log_level == "WARNING"
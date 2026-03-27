"""
Tests for configuration loading.
اختبارات تحميل الإعدادات
"""
import os
import pytest
from src.utils.config import load_config, AppConfig, _resolve_env_vars, _resolve_dict


class TestDefaults:
    """Config should return sane defaults when no file exists."""

    def test_missing_file_returns_defaults(self, tmp_dir):
        cfg = load_config(str(tmp_dir / "nonexistent.yaml"))
        assert isinstance(cfg, AppConfig)
        assert cfg.model.base_model == "microsoft/Phi-4-multimodal-instruct"
        assert cfg.model.quantization_bits == 4
        assert cfg.lora.r == 32
        assert cfg.training.epochs == 3
        assert cfg.rag.vector_store == "chromadb"

    def test_default_types(self):
        cfg = AppConfig()
        assert isinstance(cfg.model.max_length, int)
        assert isinstance(cfg.training.learning_rate, float)
        assert isinstance(cfg.lora.target_modules, list)
        assert isinstance(cfg.server.cors_origins, list)


class TestYAMLLoading:
    """Config should correctly parse YAML values."""

    def test_loads_model_config(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg.model.base_model == "test/model"
        assert cfg.model.quantization_enabled is False
        assert cfg.model.max_length == 512

    def test_loads_lora_config(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg.lora.r == 16
        assert cfg.lora.alpha == 32

    def test_loads_rag_config(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg.rag.vector_store == "chromadb"
        assert cfg.rag.top_k == 3
        assert cfg.rag.rerank is False

    def test_loads_server_config(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert cfg.server.port == 9999
        assert cfg.server.api_key == "test-key-123"
        assert cfg.server.cors_origins == ["http://localhost:3000"]


class TestEnvVarResolution:
    """Environment variable placeholders should resolve correctly."""

    def test_resolves_env_var(self):
        os.environ["TEST_VAR_ABC"] = "hello123"
        assert _resolve_env_vars("${TEST_VAR_ABC}") == "hello123"
        del os.environ["TEST_VAR_ABC"]

    def test_missing_env_var_returns_empty(self):
        result = _resolve_env_vars("${TOTALLY_NONEXISTENT_VAR_XYZ}")
        assert result == ""

    def test_non_env_string_unchanged(self):
        assert _resolve_env_vars("plain text") == "plain text"
        assert _resolve_env_vars("123") == "123"

    def test_resolve_dict_recursive(self):
        os.environ["TEST_NESTED"] = "resolved_value"
        d = {"a": {"b": "${TEST_NESTED}"}, "c": "plain"}
        result = _resolve_dict(d)
        assert result["a"]["b"] == "resolved_value"
        assert result["c"] == "plain"
        del os.environ["TEST_NESTED"]

    def test_env_var_in_config_file(self, sample_env_config_yaml):
        os.environ["TEST_API_KEY"] = "secret-from-env"
        cfg = load_config(sample_env_config_yaml)
        assert cfg.server.api_key == "secret-from-env"
        del os.environ["TEST_API_KEY"]


class TestPartialConfig:
    """Config with only some sections should use defaults for the rest."""

    def test_partial_yaml(self, tmp_dir):
        import yaml
        partial = {"model": {"base_model": "custom/model"}}
        path = tmp_dir / "partial.yaml"
        with open(path, "w") as f:
            yaml.dump(partial, f)

        cfg = load_config(str(path))
        assert cfg.model.base_model == "custom/model"
        # Other sections should be defaults
        assert cfg.lora.r == 32
        assert cfg.training.epochs == 3
        assert cfg.rag.embedding_model == "intfloat/multilingual-e5-large-instruct"

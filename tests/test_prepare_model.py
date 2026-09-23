"""Which upstream files a prepared model directory links, per weight format."""

import pytest

from tools.prepare_model import weight_files

REPO = ["config.json", "tokenizer.json", "tokenizer_config.json", "README.md",
        "model-00001-of-000002.safetensors", "model-00002-of-000002.safetensors", "model.safetensors.index.json"]


def test_safetensors_takes_every_shard_and_the_index():
    assert weight_files(REPO, "safetensors") == [
        "model-00001-of-000002.safetensors", "model-00002-of-000002.safetensors", "model.safetensors.index.json"]


def test_a_repo_without_the_requested_weights_is_refused():
    with pytest.raises(ValueError, match="no safetensors"):
        weight_files(["config.json", "tokenizer.json"], "safetensors")

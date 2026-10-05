"""The micro pipeline learns what it is shown, reloads what it saved, and repeats itself."""

import json
import math

import pytest
import torch

from micro_train.data import ByteTokenizer, load_labelled, split_text
from micro_train.model import MicroConfig, MicroTransformer
from micro_train.train import main, train_classifier, train_lm

TINY = MicroConfig(d_model=32, n_layers=1, n_heads=2, context=32)


def test_byte_tokenizer_round_trips_any_text():
    tok = ByteTokenizer()
    for text in ["", "plain ascii", "théorème (1:ℕ) + 1 ≤ 2", "emoji 🦊"]:
        assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == 256 and max(tok.encode("ÿ€")) < 256


def test_split_is_deterministic_and_disjoint():
    train, held = split_text("abcdefghij" * 10, held_out=0.2)
    assert len(train) == 80 and len(held) == 20 and train + held == "abcdefghij" * 10


def test_labels_are_indexed_in_sorted_order(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [{"text": "x", "label": "neg"}, {"text": "y", "label": "pos"},
                                                           {"text": "z", "label": "neg"}]))
    texts, labels, names = load_labelled(path)
    assert names == ["neg", "pos"] and labels == [0, 1, 0] and texts == ["x", "y", "z"]


def test_causal_lm_cannot_see_the_future():
    torch.manual_seed(0)
    model = MicroTransformer(TINY, n_classes=None).eval()
    x = torch.randint(0, 256, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 256
    assert torch.allclose(model(x)[0, :10], model(y)[0, :10], atol=1e-6)


def test_lm_learns_a_repetitive_corpus_well_below_uniform(tmp_path):
    result = train_lm("the cat sat on the mat. " * 200, TINY, steps=150, batch_size=16, lr=3e-3, seed=0, out=tmp_path)
    assert result["held_out_bits_per_byte"] < 0.5 * math.log2(256)
    assert (tmp_path / "model.safetensors").exists() and json.loads((tmp_path / "config.json").read_text())["mode"] == "lm"


def toy_rows(n: int, offset: int = 0) -> list[dict]:
    words = ["alpha", "beta", "gamma", "delta"]
    return [{"text": f"{words[i % 4]} {'good' if i % 2 else 'bad'} {words[(i * 3) % 4]}", "label": "pos" if i % 2 else "neg"}
            for i in range(offset, offset + n)]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_classifier_separates_a_trivially_separable_task(tmp_path):
    train = write_rows(tmp_path / "train.jsonl", toy_rows(200))
    held = write_rows(tmp_path / "eval.jsonl", toy_rows(50, offset=1000))
    result = train_classifier(train, held, TINY, steps=150, batch_size=16, lr=3e-3, seed=0, out=tmp_path / "clf")
    assert result["eval_accuracy"] >= 0.95 and result["labels"] == ["neg", "pos"]


def test_same_seed_same_metrics(tmp_path):
    runs = [train_lm("abcabcabd " * 100, TINY, steps=20, batch_size=8, lr=3e-3, seed=7, out=tmp_path / str(i)) for i in range(2)]
    assert runs[0]["held_out_bits_per_byte"] == runs[1]["held_out_bits_per_byte"]


def test_cli_writes_metrics_and_refuses_an_empty_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world. " * 100)
    main(["lm", "--data", str(corpus), "--out", str(tmp_path / "run"), "--steps", "5", "--d-model", "32",
          "--layers", "1", "--heads", "2", "--context", "32"])
    assert json.loads((tmp_path / "run" / "metrics.json").read_text())["steps"] == 5
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(SystemExit):
        main(["lm", "--data", str(tmp_path / "empty.txt"), "--out", str(tmp_path / "x")])

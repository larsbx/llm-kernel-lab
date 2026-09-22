"""The loop: sample k, verify each distinct proof once, and record what the kernel said."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prover_loop.client import Completion, sampler
from prover_loop.problems import Problem
from prover_loop.run import run
from prover_loop.verify import Verdict

A = Problem("a", "", "theorem a : True := by\n")
B = Problem("b", "", "theorem b : False := by\n")


def fake_sample(completions):
    return lambda prompt, k: [Completion(text, 3, 0.5) for text in completions[prompt][:k]]


def test_run_records_every_attempt_and_only_verified_proofs(tmp_path):
    from prover_loop.prompt import prompt
    sample = fake_sample({prompt(A): ["  simp\n```", "  trivial\n```", "  trivial\n```"],
                          prompt(B): ["  sorry\n```", "  exact?\n```", "  sorry\n```"]})
    calls = []

    def verify(source, theorem):
        calls.append(source)
        return Verdict(source.endswith("trivial\n"), "verified" if source.endswith("trivial\n") else "lean error", 1.0)

    summary = run([A, B], sample, verify, k=3, out=tmp_path)

    assert summary["solved"] == ["a"] and summary["pass_at_k"] == 0.5 and summary["attempts"] == 6
    assert len(calls) == 4, "each distinct proof is checked once"
    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [(r["problem"], r["sample"], r["verified"]) for r in attempts] == [
        ("a", 0, False), ("a", 1, True), ("a", 2, True), ("b", 0, False), ("b", 1, False), ("b", 2, False)]
    verified = [json.loads(line) for line in (tmp_path / "verified.jsonl").read_text().splitlines()]
    assert verified == [{"problem": "a", "prompt": prompt(A), "completion": "  trivial\n```"}]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


class Stub(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append(body)
        reply = {"choices": [{"text": f"  exact h{body['seed']}\n```"}], "usage": {"completion_tokens": 4}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_sampler_sends_k_seeded_requests_to_an_openai_completions_endpoint():
    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sample = sampler(f"http://127.0.0.1:{server.server_port}", "m", temperature=1.0, top_p=0.95, max_tokens=64, seed=10)
        out = sample("P", 3)
    finally:
        server.shutdown()
    assert sorted(c.text for c in out) == ["  exact h10\n```", "  exact h11\n```", "  exact h12\n```"]
    assert all(c.tokens == 4 for c in out)
    assert {(b["model"], b["prompt"], b["stop"][0], b["max_tokens"]) for b in Stub.seen} == {("m", "P", "```", 64)}

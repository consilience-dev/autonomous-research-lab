"""Exercise the Run 2 CLI ledger with a synthetic model and temporary work."""
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent-run2"))
import lab


class BudgetTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.work = self.root / "work"
        self.calls = self.work / "calls"
        self.calls.mkdir(parents=True)
        (self.root / "source.json").write_text("{}")
        self.inputs = self.root / "inputs.jsonl"
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=36, max_position_embeddings=32768),
            generation_config=SimpleNamespace(eos_token_id=2),
            generate=lambda **kwargs: self.generate(**kwargs),
        )
        self.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "synthetic")
        self.failure = False

    def generate(self, **kwargs):
        self.assertEqual(kwargs["max_new_tokens"], 512)
        if self.failure:
            raise KeyboardInterrupt
        ids = kwargs["input_ids"]
        return torch.cat((ids, torch.ones((1, 512), dtype=ids.dtype)), dim=1)

    def run_lab(self, *args, code=0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (patch.object(sys, "argv", ["lab", *args]),
              patch.object(lab, "Weights", return_value=SimpleNamespace(base=self.root, work=self.work)),
              patch.object(lab, "model_and_tokenizer", return_value=(self.model, self.tokenizer)),
              patch.object(lab, "encode", return_value=torch.tensor([[1, 1]])),
              patch.object(lab, "measure", return_value={"hidden.0": torch.zeros(2, 1)}),
              redirect_stdout(stdout), redirect_stderr(stderr)):
            result = lab.main(self.root, self.work)
        self.assertEqual(result, code, stderr.getvalue())
        return json.loads(stdout.getvalue()) if code == 0 else stderr.getvalue()

    def request(self, command, n, *args, code=0):
        self.inputs.write_text('"example"\n' * n)
        extra = ("--hidden", "0") if command == "measure" else ()
        return self.run_lab(command, str(self.inputs), *extra, *args, code=code)

    def seed(self, command, examples, confirmation=False):
        path = self.calls / f"seed-{len(list(self.calls.iterdir()))}"
        path.mkdir()
        (path / "charge.json").write_text(json.dumps(
            {"command": command, "examples": examples, "confirmation": confirmation}))

    def test_pools_and_cutoff(self):
        self.seed("generate", 120)
        self.seed("generate", 40, confirmation=True)
        self.assertIn("1 to 8", self.request("generate", 9, "--paired", "--revision", "edit", code=1))
        result = self.request("generate", 8, "--paired", "--revision", "edit")
        rows = [json.loads(line) for line in Path(result["examples_file"]).read_text().splitlines()]
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r["length_cutoff"] and len(r["output_ids"]) == 512 for r in rows))
        self.assertIn("exploration", self.request("generate", 1, code=1))
        self.request("generate", 8, "--paired", "--revision", "edit", "--confirmation")
        self.assertIn("confirmation", self.request("generate", 1, "--confirmation", code=1))
        self.seed("measure", 112)
        self.assertIn("1 to 16", self.request("measure", 17, code=1))
        self.request("measure", 16)
        self.assertIn("measurement", self.request("measure", 1, code=1))
        status = self.run_lab("status")
        self.assertEqual(status["completed_passes"], 3)
        self.assertEqual(status["charged_examples"], {"generate": 192, "confirmation": 56, "measure": 128})
        self.assertTrue(all(left == 0 for left in status["remaining_examples"].values()))

    def test_completed_call_limit_and_interrupted_charge(self):
        for i in range(49):
            path = self.calls / f"completed-{i}"
            path.mkdir()
            (path / "pass.completed").touch()
        self.failure = True
        self.assertIn("KeyboardInterrupt", self.request("generate", 2, "--confirmation", code=1))
        status = self.run_lab("status")
        self.assertEqual(status["completed_passes"], 49)
        self.assertEqual(status["remaining_examples"], {
            "generation_exploration": 136, "generation_confirmation": 54, "measurement": 128,
        })
        self.failure = False
        self.request("generate", 1)
        self.assertIn("Forward-pass budget exhausted", self.request("generate", 1, "--confirmation", code=1))
        self.assertIn("Forward-pass budget exhausted", self.request("measure", 1, code=1))
        status = self.run_lab("status")
        self.assertEqual(status["remaining_passes"], 0)
        self.assertEqual(status["charged_examples"]["generate"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

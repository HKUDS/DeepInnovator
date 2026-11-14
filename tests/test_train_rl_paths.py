"""Exercise the launcher's path setup without invoking Ray or training."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "recipe/DeepInnovator/train_rl.sh"
MARKER = "# End launcher path setup."


class LauncherPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repository with spaces"
        self.script_dir = self.root / "recipe" / "DeepInnovator"
        (self.script_dir / "config").mkdir(parents=True)
        (self.root / "verl").mkdir()
        (self.root / "verl" / "__init__.py").write_text("# synthetic package\n", encoding="utf-8")
        self.files = [
            self.script_dir / "config" / "agent.yaml",
            self.script_dir / "config" / "reward_config.yaml",
            self.script_dir / "config" / "ResearchGAN_interaction_config.yaml",
            self.script_dir / "reward_function.py",
        ]
        for path in self.files:
            path.touch()
        prefix, marker, _ = SOURCE.read_text(encoding="utf-8").partition(MARKER)
        self.assertTrue(marker, "launcher setup boundary is missing")
        self.script = self.script_dir / "probe.sh"
        self.script.write_text(
            prefix + marker + "\n"
            + "printf '%s\\n' \"$PWD\" \"$PROJECT_DIR\" \"$AGENTLOOP_CONFIG_PATH\" "
            + "\"$RewardConfigPath\" \"$RewardFunctionPath\" \"$INTERACTION_CONFIG_PATH\" "
            + "\"$PYTHONPATH\" \"${1:-}\"\n",
            encoding="utf-8",
        )

    def run_probe(self, cwd):
        env = {**os.environ, "PYTHONPATH": "existing-python-path"}
        return subprocess.run(
            ["bash", str(self.script), "relative/checkpoint"], cwd=cwd,
            text=True, capture_output=True, env=env, check=False,
        )

    def test_documented_and_other_cwd_preserve_caller_paths(self):
        for cwd in (self.script_dir, self.root):
            with self.subTest(cwd=cwd):
                result = self.run_probe(cwd)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [
                    str(cwd), str(self.root), str(self.files[0]), str(self.files[1]),
                    str(self.files[3]), str(self.files[2]),
                    f"{self.root}:existing-python-path", "relative/checkpoint",
                ])
                env = {**os.environ, "PYTHONPATH": result.stdout.splitlines()[6]}
                probe = subprocess.run(
                    [sys.executable, "-c", "import importlib.util; print(importlib.util.find_spec('verl').origin)"],
                    cwd=cwd, text=True, capture_output=True, env=env, check=True,
                )
                self.assertEqual(probe.stdout.strip(), str(self.root / "verl" / "__init__.py"))

    def test_missing_required_configuration_fails_before_training(self):
        self.files[2].unlink()
        result = self.run_probe(self.script_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required launcher file:", result.stderr)
        self.assertIn("ResearchGAN_interaction_config.yaml", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_missing_bundled_framework_fails_before_training(self):
        (self.root / "verl" / "__init__.py").unlink()
        (self.root / "verl").rmdir()
        result = self.run_probe(self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing bundled verl directory:", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()

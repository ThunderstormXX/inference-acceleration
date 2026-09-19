"""Exercise setup plans without installing packages or touching live environments."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_setup(name):
    spec = importlib.util.spec_from_file_location(f"setup_{name}_test", ROOT / "scripts/setup" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupReproducibilityTests(unittest.TestCase):
    def test_all_setups_use_pinned_python_and_preserve_locks(self):
        cases = [
            ("environment", "mac", "requirements-mac.lock", "3.13.5"),
            ("environment", "vllm", "requirements-vllm.lock", "3.12.8"),
            ("transformers", None, "requirements-transformers.lock", "3.13.5"),
        ]
        for name, backend, lock_name, expected_python in cases:
            with self.subTest(backend=backend or name), tempfile.TemporaryDirectory() as temporary:
                module = load_setup(name)
                root = Path(temporary)
                lock = root / lock_name
                original = "example==1.2.3\n"
                lock.write_text(original)
                with patch.object(module, "ROOT", root), \
                        patch.object(module.platform, "system", return_value="Darwin"), \
                        patch.object(module.platform, "machine", return_value="arm64"), \
                        patch.object(module.shutil, "which", return_value="/fake/uv"), \
                        patch.object(module.subprocess, "run") as run, \
                        patch.object(module.subprocess, "check_output", return_value=expected_python + "\n"):
                    if backend:
                        module.EnvironmentSetup(backend).run()
                    else:
                        module.TransformersEnvironmentSetup(prewarm_kernel=False).run()
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(commands[0][1:4], ["venv", "--python", expected_python])
                self.assertEqual(commands[1][1:3], ["pip", "sync"])
                self.assertEqual(commands[1][-1], str(lock))
                self.assertEqual(commands[2][1:3], ["pip", "check"])
                self.assertEqual(lock.read_text(), original)

    def test_missing_lock_does_not_create_environment(self):
        module = load_setup("environment")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(module, "ROOT", Path(temporary)), \
                patch.object(module.platform, "system", return_value="Darwin"), \
                patch.object(module.platform, "machine", return_value="arm64"), \
                patch.object(module.shutil, "which", return_value="/fake/uv"), \
                patch.object(module.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "lock is missing"):
                module.EnvironmentSetup("mac").run()
            run.assert_not_called()

    def test_wrong_existing_python_does_not_install_packages(self):
        module = load_setup("environment")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "requirements-mac.lock").write_text("example==1.2.3\n")
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(module, "ROOT", root), \
                    patch.object(module.platform, "system", return_value="Darwin"), \
                    patch.object(module.platform, "machine", return_value="arm64"), \
                    patch.object(module.shutil, "which", return_value="/fake/uv"), \
                    patch.object(module.subprocess, "run") as run, \
                    patch.object(module.subprocess, "check_output", return_value="3.13.99\n"):
                with self.assertRaisesRegex(RuntimeError, "requires 3.13.5"):
                    module.EnvironmentSetup("mac").run()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

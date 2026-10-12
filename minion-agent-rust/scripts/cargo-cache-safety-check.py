"""In-memory refusal controls: no filesystem mutation reaches the OS."""
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('safety', Path(__file__).with_name('cargo-cache-safety.py'))
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)
attempts = []
for value in ('', '/target', str(safety.PROJECT / '.tmp/cache/cargo-target/windows'), '../controls', 'C:/outside'):
    with patch.dict(os.environ, {'MINION_CONTROL_CARGO_TARGET': value}), patch.object(Path, 'mkdir', side_effect=lambda *a, **k: attempts.append(a)):
        try:
            safety.control_target()
        except RuntimeError:
            pass
        else:
            raise AssertionError(f'unsafe target accepted: {value}')
assert not attempts
print('CACHE SAFETY REFUSAL CONTROLS PASS 5/5, zero mutation calls')

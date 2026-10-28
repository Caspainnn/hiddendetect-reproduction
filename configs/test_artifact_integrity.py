"""Ensure corrupted or truncated downloads cannot become usable model files."""
import hashlib
from pathlib import Path
import tempfile
from fetch_locked import verify

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'weight'
    expected = hashlib.sha256(b'valid').hexdigest()
    assert not verify(path, 5, expected)
    path.write_bytes(b'valid')
    assert verify(path, 5, expected)
    path.write_bytes(b'wrong')
    assert not verify(path, 5, expected)
    path.write_bytes(b'val')
    assert not verify(path, 5, expected)
print('PASS: missing, valid, same-length corruption, truncation')

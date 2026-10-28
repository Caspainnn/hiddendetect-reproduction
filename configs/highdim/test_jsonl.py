import json
import tempfile
from pathlib import Path
from features import read_jsonl

with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'records.jsonl'
    rows=[{'text':'a\u2028b\u0085c\u2029d'},{'text':'ordinary\nnewline'}]
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    assert read_jsonl(path)==rows
print('PASS: Unicode separators inside JSON strings are not record delimiters')

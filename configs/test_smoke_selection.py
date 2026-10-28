"""Contract: deterministic, balanced, calibrated-prompt exclusions, fail closed."""
from prepare_smoke import select_smoke

rows = [{'id': str(i), 'type': 'safe' if i < 8 else 'contrast_test', 'prompt': f'prompt {i}'} for i in range(16)]
selected = select_smoke(rows, {'prompt 0'})
assert selected == select_smoke(list(reversed(rows)), {'prompt 0'})
assert len(selected) == 8
assert sum(x['label'] for x in selected) == 4
assert '0' not in {x['id'] for x in selected}
assert all(len(x['prompt_sha256']) == 64 for x in selected)
try:
    select_smoke(rows[:8], set())
except ValueError:
    pass
else:
    raise AssertionError('Must reject missing unsafe class')
try:
    select_smoke(rows + [rows[0]], set())
except ValueError:
    pass
else:
    raise AssertionError('Must reject duplicate IDs')
print('PASS: stable order, 4+4 balance, calibration exclusion, missing class and duplicate ID checks')

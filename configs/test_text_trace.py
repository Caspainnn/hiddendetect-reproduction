"""Catch shifted layer intervals and missing/extra captures without a model."""
from run_text import validate_traces

traces = [[1.0] * 32, list(map(float, range(32)))]
assert validate_traces(traces, [13.0, 292.5]) is None
for bad_traces, bad_scores in [(traces, [14.0, 292.5]), ([traces[0][:31]], [13.0]),
                                (traces, [13.0]), ([[float('nan')] * 32], [0.0])]:
    try:
        validate_traces(bad_traces, bad_scores)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid trace/score accepted')
print('PASS: closed interval, 32 layers, sample count, finite values')

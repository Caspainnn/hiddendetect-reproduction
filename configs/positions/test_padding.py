from diagnose_prefix import padded
assert padded([1,2,3],5,0)==([1,2,3,0,0],[1,1,1,0,0])
try:padded([1,2,3],2,0)
except AssertionError:pass
else:raise AssertionError('Padding truncated input')
print('PASS: diagnostic padding preserves valid input and marks only padding')

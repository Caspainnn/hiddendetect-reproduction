from resume import execution_phase,merge_cost
assert execution_phase({0,1,2},{0,1},True)=='reuse_sealed'
assert execution_phase({0},{0,1},False)=='collect_train'
try:execution_phase({0,1,2},{0,1},False)
except AssertionError:pass
else:raise AssertionError('unsealed heldout access accepted')
cost=merge_cost({'counts':{'a':4},'seconds':{'a':1.5},'maximum_allocated_gpu_bytes':9},{'counts':{'a':3,'b':1},'seconds':{'a':2.,'b':.5},'maximum_allocated_gpu_bytes':8})
assert cost=={'counts':{'a':7,'b':1},'seconds':{'a':3.5,'b':.5},'maximum_allocated_gpu_bytes':9}
print('SEALED_RECOVERY_AND_CUMULATIVE_COST_CONTRACT_PASS')

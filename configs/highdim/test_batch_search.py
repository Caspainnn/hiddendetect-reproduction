import sys
import numpy as np
from features import ROOT
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from experiment_core import binary_pso
from batch_search import batched_binary_pso
def objective(mask): return float(mask[0])-.03*mask.sum()
for seed in (0,1,2):
    for budget in (5,20,100):
        old=binary_pso(6,objective,seed,budget=budget,particles=10,rounds=10)
        new=batched_binary_pso(6,lambda masks:[objective(x) for x in masks],seed,budget=budget,particles=10,rounds=10)
        assert old==new,(seed,budget,old,new)
print('PASS: batching preserves exact serial PSO masks, fitness and unique-evaluation order')

import pandas as pd
import numpy as np
from utils import getNextVer

files = [
    "/Users/ed/Git/kaggle/credit-crisis/solution2/output-lgb-ordinal-5-0.7233.txt",
    "/Users/ed/Git/kaggle/credit-crisis/solution2/output-lr-ordinal-1-0.6282.txt",
    "/Users/ed/Git/kaggle/credit-crisis/solution2/output-rf-ordinal-3-0.7142.txt",
    "/Users/ed/Git/kaggle/credit-crisis/solution2/output-xgb-ordinal-4-0.7196.txt",
]
output = pd.read_csv(files[0], header=None, sep='\t')
output.columns = ['id', 'prob']

for file in files[1:]:
    x =  pd.read_csv(file, header=None, sep='\t')
    x.columns = ['id', 'prob']
    output['prob'] += x['prob']

output['prob'] /= len(files)
output.to_csv('output-merged-{}.txt'.format(getNextVer("output-merged-(\d+).txt")),
              index=False, header=False, sep='\t')



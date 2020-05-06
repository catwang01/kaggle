import pandas as pd

import numpy as np

x1 =  pd.read_csv('/Users/ed/Git/kaggle/credit-crisis/solution2/output-rf-ordinal-1-0.7233.txt', header=None, sep='\t')
x1.columns = ['id', 'prob']
x2 =  pd.read_csv('/Users/ed/Git/kaggle/credit-crisis/solution2/output-xgb-ordinal-1-0.7241.txt', header=None, sep='\t')
x2.columns = ['id', 'prob']
output = pd.DataFrame({'id': x1.id, 'prob': (x1['prob'] + x2['prob']) / 2})
output.to_csv('output-merged.txt', index=False, header=False, sep='\t')


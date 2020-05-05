# EDA
tagtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_tag.csv'
behtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_beh.csv'
trdtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_trd.csv'

tagtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_tag.csv'
behtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_beh.csv'
trdtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_trd.csv'

### 1. beh_df 共 4个feature
import pandas as pd

beh_df = pd.read_csv(behtrainPath)
print(list(beh_df.columns))

### 3. tag 共 43 个Feature
import pandas as pd

tag_df = pd.read_csv(tagtrainPath)
features = ['id', 'gdr_cd', 'age', 'mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd', 'job_year', 'ic_ind',
                'fr_or_sh_ind', 'dnl_mbl_bnk_ind', 'dnl_bind_cmb_lif_ind', 'hav_car_grp_ind', 'hav_hou_grp_ind',
                'l6mon_agn_ind', 'frs_agn_dt_cnt', 'vld_rsk_ases_ind', 'fin_rsk_ases_grd_cd',
                'confirm_rsk_ases_lvl_typ_cd', 'cust_inv_rsk_endu_lvl_cd', 'l6mon_daim_aum_cd', 'tot_ast_lvl_cd',
                'pot_ast_lvl_cd', 'bk1_cur_year_mon_avg_agn_amt_cd', 'l12mon_buy_fin_mng_whl_tms',
                'l12_mon_fnd_buy_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms', 'loan_act_ind',
                'pl_crd_lmt_cd', 'ovd_30d_loan_tot_cnt', 'his_lng_ovd_day', 'hld_crd_card_grd_cd', 'crd_card_act_ind',
                'l1y_crd_card_csm_amt_dlm_cd', 'atdd_type', 'perm_crd_lmt_cd', 'cur_debit_cnt', 'cur_credit_cnt',
                'cur_debit_min_opn_dt_cnt', 'cur_credit_min_opn_dt_cnt', 'cur_debit_crd_lvl']

tag_df = tag_df[features]
print(len(tag_df.columns))
print(list(tag_df.columns))

# 融合 to_csv
beh_df = beh_df.merge(tag_df, on='id', how='left')

# 检查有无null #num_beh_df[beh_df.describe().isnull().values] features = list(beh_df.describe())
num_beh_df = beh_df[features]
num_beh_df.to_csv('num_beh_df.csv', index=False)

### 2. trd_df 共 8 个feature - ['id', 'flag', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd', 'trx_tm', 'cny_trx_amt']

import pandas as pd

trd_df = pd.read_csv(trdtrainPath)
print(list(trd_df.columns))

# 融合 to_csv trd_df = trd_df.merge(tag_df,on='id',how='left')

# 检查有无Nan #num_trd_df[trd_df.describe().isnull().values]
num_trd_df = trd_df[list(trd_df.describe())]
num_trd_df.to_csv('num_trd_df.csv', index=False)

# 评分数据 - 评分数据集_beh_a.csv

## 测试数据集 - 评分数据集_beh_a.csv

import pandas as pd

pf_beh_df = pd.read_csv(behtestPath)
pf_trd_df = pd.read_csv(trdtestPath)
pf_tag_df = pd.read_csv(tagtestPath)
print(len(pf_beh_df))

print(len(set(pf_beh_df['id'])))

features = ['id', 'gdr_cd', 'age', 'mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd', 'job_year', 'ic_ind',
                'fr_or_sh_ind', 'dnl_mbl_bnk_ind', 'dnl_bind_cmb_lif_ind', 'hav_car_grp_ind', 'hav_hou_grp_ind',
                'l6mon_agn_ind', 'frs_agn_dt_cnt', 'vld_rsk_ases_ind', 'fin_rsk_ases_grd_cd',
                'confirm_rsk_ases_lvl_typ_cd', 'cust_inv_rsk_endu_lvl_cd', 'l6mon_daim_aum_cd', 'tot_ast_lvl_cd',
                'pot_ast_lvl_cd', 'bk1_cur_year_mon_avg_agn_amt_cd', 'l12mon_buy_fin_mng_whl_tms',
                'l12_mon_fnd_buy_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms', 'loan_act_ind',
                'pl_crd_lmt_cd', 'ovd_30d_loan_tot_cnt', 'his_lng_ovd_day', 'hld_crd_card_grd_cd', 'crd_card_act_ind',
                'l1y_crd_card_csm_amt_dlm_cd', 'atdd_type', 'perm_crd_lmt_cd', 'cur_debit_cnt', 'cur_credit_cnt',
                'cur_debit_min_opn_dt_cnt', 'cur_credit_min_opn_dt_cnt', 'cur_debit_crd_lvl']

pf_tag_df = pf_tag_df[features]
print(len(pf_tag_df.columns))
print(list(pf_tag_df.columns))

# 融合 to_csv
pf_beh_df = pf_beh_df.merge(pf_tag_df, on='id', how='left')

# 检查有无null #num_beh_df[beh_df.describe().isnull().values] features = list(pf_beh_df.describe())

num_pf_beh_df = pf_beh_df[features]
num_pf_beh_df.to_csv('num_pf_beh_df.csv', index=False)

# 融合 to_csv pf_trd_df = pf_trd_df.merge(pf_tag_df,on='id',how='left')
# 检查有无Nan #num_trd_df[trd_df.describe().isnull().values]
num_pf_trd_df = pf_trd_df[list(pf_trd_df.describe())]
num_pf_trd_df.to_csv('num_pf_trd_df.csv', index=False)

num_pf_trd_df.head()

import pandas as pd

num_beh_df = pd.read_csv('num_beh_df.csv')
num_beh_df.head()

num_trd_df = pd.read_csv('num_trd_df.csv')
num_trd_df.head()

# from matplotlib import pyplot as plt
# import seaborn as sns
#
# f, ax = plt.subplots(1, 2, figsize=(10, 4))
# num_beh_df['flag'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
# ax[0].set_title('flag')
# ax[0].set_ylabel('')
# sns.countplot('flag', data=num_beh_df, ax=ax[1])
# ax[1].set_title('flag')
# plt.show()

# 模型
# importing all the required ML packages from sklearn.linear_model import LogisticRegression #logistic regression from sklearn import svm #support vector Machine from sklearn.ensemble import RandomForestClassifier #Random Forest from sklearn.neighbors import KNeighborsClassifier #KNN from sklearn.naive_bayes import GaussianNB #Naive bayes from sklearn.tree import DecisionTreeClassifier #Decision Tree from sklearn.model_selection import train_test_split #training and testing data split from sklearn import metrics #accuracy measure from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = num_trd_df

train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['flag'])
train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]
X = data[data.columns[1:]]
Y = data['flag']

model = LogisticRegression()
model.fit(train_X, train_Y)
prediction3 = model.predict(test_X)
print('The accuracy of the Logistic Regression is', accuracy_score(prediction3, test_Y))

## 预测

import pandas as pd

pf_beh_df = pd.read_csv(behtestPath)
pf_trd_df = pd.read_csv(trdtestPath)
pf_tag_df = pd.read_csv(tagtestPath)
print(len(pf_trd_df))

pf_trd_df_id_list = list(pf_trd_df['id'])
num_pf_trd_df = pd.read_csv('num_pf_trd_df.csv')
num_pf_trd_df['id'] = pf_trd_df_id_list

num_pf_trd_df.head()

data = num_pf_trd_df

print(len(data))
X = data[data.columns[:-1]]

# 直接预测 0,1
prediction = model.predict_proba(X)
prediction_list = list(prediction[:, 1])

count = 1
for i in prediction[:, 1]:
    if i > 0.5:
        count += 1
print(count)

from pandas import DataFrame


result_pre_df = DataFrame({'id': pf_trd_df_id_list, 'prob': prediction_list})
print(len(result_pre_df))
result_pre_df.head()

result_pre_df.to_csv('result_pre.csv')

output = result_pre_df.groupby('id').mean()
output = output.reset_index()


count_wy_dict = dict()
for item in set(data['id']):
    count_wy_dict.update({item[1:]: []})

for ids, prob in zip(result_pre_df['id'], result_pre_df['prob']):
    count_wy_dict[ids[1:]].append(prob)

keys = list(count_wy_dict.keys())

#keys = map(lambda x: int(x), keys)

with open('upload_1.txt', 'w', encoding='utf-8') as f:
    for k in sorted(list(keys)):
        pred = sum(count_wy_dict[str(k)]) / len(count_wy_dict[str(k)])
        f.write('U' + str(k) + '\t' + str(pred) + '\n')

result_pre_df.to_csv('result_pre.csv')

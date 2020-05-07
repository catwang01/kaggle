from baseModel import Trainer
from sklearn.ensemble import RandomForestClassifier
from matplotlib import  pyplot as plt
from path import *
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

other_params = {
    'n_jobs': -1,
    'n_estimators': 20,
    'max_depth': 12,
    'random_state': 4
}

tuned_params = {
    'n_estimators': [10, 20, 30, 40, 50, 60],
    'max_depth': [8, 9, 10, 12, 15, 20],
    # 'random_state': [1,2,3,4,5]
}

trainer = Trainer(modelClass=RandomForestClassifier,
                  other_params=other_params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=True,
                  modelName='rf',
                  cv = 5,
                  dataPath=processedDataPath)

trainer.read_data()
trainer.fit()
output = trainer.getOutput()

bestResult = pd.read_csv('currentBest.txt', header=None, sep='\t')
bestResult.columns = ['id', 'prob']

pccs = pearsonr(output['prob'], bestResult['prob'])
print("Score prediction: {}".format(pccs))

sorted_importances, sorted_featurenames= zip(*sorted(zip(trainer.model.feature_importances_, trainer.feature_names), reverse=True))
plt.plot(sorted_importances)
plt.title("变量累计贡献")
plt.show()
plt.plot(np.cumsum(sorted_importances))
plt.title("累计贡献")
plt.show()
sorted_featurenames[:10]
sorted_featurenames


# 感觉 cny不需要分很细
# page_no 也不需要
# l12_mon_tms 貌似也一般
# ('gdr_cd',
#  'hav_car_grp_ind',
#  'hld_crd_card_grd_cd',
#  'cny_trx_amt_CB03',
#  'acdm_deg_cd',
#  'l1y_crd_card_csm_amt_dlm_cd',
#  'cur_debit_crd_lvl',
#  'newage',
#  'cny_trx_amt_BB01',
#  'dnl_bind_cmb_lif_ind',
#  'cur_credit_min_opn_dt_cnt',
#  'new_job_year',
#  'atdd_type',
#  'mrg_situ_cd',
#  'cur_debit_min_opn_dt_cnt',
#  'cur_debit_cnt',
#  'fr_or_sh_ind',
#  'dnl_mbl_bnk_ind',
#  'perm_crd_lmt_cd',
#  'edu_deg_cd',
#  'cur_credit_cnt',
#  'deg_cd',
#  'vld_rsk_ases_ind',
#  'cny_trx_amt_CA02',
#  'crd_card_act_ind',
#  'pot_ast_lvl_cd',
#  'fin_rsk_ases_grd_cd',
#  'bk1_cur_year_mon_avg_agn_amt_cd',
#  'cny_trx_amt_BA03',
#  'confirm_rsk_ases_lvl_typ_cd',
#  'cny_trx_amt_CB02',
#  'has_credit_but_not_debit',
#  'page_no_CQA',
#  'tot_ast_lvl_cd',
#  'cny_trx_amt_BB03',
#  'cny_trx_amt_CA03',
#  'l6mon_agn_ind',
#  'cny_trx_amt_BA01',
#  'page_no_CQE',
#  'loan_act_ind',
#  'page_no_AAO',
#  'page_no_XAI',
#  'page_no_MSG',
#  'page_no_TRN',
#  'pl_crd_lmt_cd',
#  'page_no_FTR',
#  'page_no_SZA',
#  'page_no_SZD',
#  'page_no_XAG',
#  'page_no_BWA',
#  'page_no_CQD',
#  'page_no_BWE',
#  'page_no_CQB',
#  'page_no_CQC',
#  'page_no_GBA',
#  'page_no_CTR',
#  'cny_trx_amt_CC03',
#  'ic_ind',
#  'page_no_EGA',
#  'cny_trx_amt_CC02',
#  'cust_inv_rsk_endu_lvl_cd',
#  'l12_mon_tms',
#  'hav_hou_grp_ind',
#  'cny_trx_amt_BC03',
#  'page_no_LC0',
#  'page_no_JF2',
#  'page_no_FLS',
#  'page_no_EGB',
#  'cny_trx_amt_BC01',
#  'page_no_SYK',
#  'page_no_LCT',
#  'page_no_FDA',
#  'page_no_MTA',
#  'page_no_ZY1',
#  'l6mon_daim_aum_cd',
#  'page_no_JJK',
#  'page_no_JJD',
#  'cny_trx_amt_CC01',
#  'cny_trx_amt_CB01',
#  'cny_trx_amt_CA01',
#  'cny_trx_amt_BC02',
#  'cny_trx_amt_BB02',
#  'cny_trx_amt_BA02')
#

from baseModel import Trainer
from sklearn.ensemble import RandomForestClassifier
from matplotlib import  pyplot as plt
from path import *
import numpy as np

other_params = {
    'n_jobs': -1,
    'n_estimators': 30,
    'max_depth': 12,
}

tuned_params = {
    'n_estimators': [10, 15, 20, 25, 30],
    'max_depth': [10, 15, 20, 25]
}

trainer = Trainer(modelClass=RandomForestClassifier,
                  other_params=other_params,
                  tuned_params=tuned_params,
                  isupdate=True,
                  istune=False,
                  modelName='rf',
                  dataPath=processedDataPath)

trainer.read_data()
trainer.fit()
trainer.getOutput()
sorted_importances, sorted_featurenames= zip(*sorted(zip(trainer.model.feature_importances_, trainer.feature_names), reverse=True))
plt.plot(sorted_importances)
plt.title("变量累计贡献")
plt.show()
plt.plot(np.cumsum(sorted_importances))
plt.title("累计贡献")
plt.show()
sorted_featurenames[:10]
sorted_featurenames

# Out[35]:
# ('cny_trx_amt_CB03',
#  'cny_trx_amt_BB01',
#  'l1y_crd_card_csm_amt_dlm_cd',
#  'cny_trx_amt_BB03',
#  'cny_trx_amt_CA02',
#  'cny_trx_amt_BA03',
#  'cny_trx_amt_CB02',
#  'perm_crd_lmt_cd',
#  'acdm_deg_cd',
#  'edu_deg_cd',
#  'gdr_cd',
#  'cur_credit_min_opn_dt_cnt',
#  'page_no_CQA',
#  'crd_card_act_ind',
#  'cny_trx_amt_BA01',
#  'hav_car_grp_ind',
#  'cny_trx_amt_CA03',
#  'pot_ast_lvl_cd',
#  'dnl_bind_cmb_lif_ind',
#  'page_no_CQE',
#  'cur_debit_cnt',
#  'cur_credit_cnt',
#  'hld_crd_card_grd_cd',
#  'fr_or_sh_ind',
#  'page_no_AAO',
#  'newage',
#  'mrg_situ_cd',
#  'confirm_rsk_ases_lvl_typ_cd',
#  'cur_debit_crd_lvl',
#  'cur_debit_min_opn_dt_cnt',
#  'page_no_MSG',
#  'tot_ast_lvl_cd',
#  'page_no_XAI',
#  'page_no_SZA',
#  'page_no_TRN',
#  'atdd_type',
#  'page_no_FTR',
#  'deg_cd',
#  'dnl_mbl_bnk_ind',
#  'fin_rsk_ases_grd_cd',
#  'page_no_CQD',
#  'cny_trx_amt_CC02',
#  'page_no_XAG',
#  'bk1_cur_year_mon_avg_agn_amt_cd',
#  'cust_inv_rsk_endu_lvl_cd',
#  'page_no_SZD',
#  'new_job_year',
#  'page_no_BWA',
#  'page_no_CTR',
#  'vld_rsk_ases_ind',
#  'cny_trx_amt_CC03',
#  'page_no_GBA',
#  'page_no_CQB',
#  'l6mon_agn_ind',
#  'page_no_BWE',
#  'page_no_CQC',
#  'page_no_EGA',
#  'cny_trx_amt_BC03',
#  'loan_act_ind',
#  'cny_trx_amt_BC01',
#  'pl_crd_lmt_cd',
#  'page_no_LC0',
#  'has_credit_but_not_debit',
#  'ic_ind',
#  'page_no_EGB',
#  'page_no_JF2',
#  'hav_hou_grp_ind',
#  'l12_mon_tms',
#  'page_no_FLS',
#  'page_no_ZY1',
#  'page_no_SYK',
#  'page_no_MTA',
#  'page_no_LCT',
#  'l6mon_daim_aum_cd',
#  'page_no_FDA',
#  'page_no_JJK',
#  'page_no_JJD',
#  'cny_trx_amt_CC01',
#  'cny_trx_amt_CB01',
#  'cny_trx_amt_CA01',
#  'cny_trx_amt_BC02',
#  'cny_trx_amt_BB02',
#  'cny_trx_amt_BA02')

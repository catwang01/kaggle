import os

root = os.path.dirname(__file__)
processed_train_trd_feature_path = os.path.join(root, 'processed_train_trd_feature.pkl')
processed_test_trd_feature_path = os.path.join(root, 'processed_test_trd_feature.pkl')
processed_train_tag_feature_path = os.path.join(root, 'processed_train_tag_feature.pkl')
processed_test_tag_feature_path = os.path.join(root, 'processed_test_tag_feature.pkl')
jsonPath = os.path.join(root, "colNames.json")

tagtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_tag.csv'
behtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_beh.csv'
trdtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_trd.csv'

tagtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_tag.csv'
behtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_beh.csv'
trdtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_trd.csv'

import os

root = os.path.dirname(__file__)
# processed_train_trd_feature_path = os.path.join(root, 'processed_train_trd_feature.pkl')
# processed_test_trd_feature_path = os.path.join(root, 'processed_test_trd_feature.pkl')
# processed_train_tag_feature_path = os.path.join(root, 'processed_train_tag_feature.pkl')
# processed_test_tag_feature_path = os.path.join(root, 'processed_test_tag_feature.pkl')
# pcaTrainFeaturePath = os.path.join(root, 'pcaTrainFeature.pkl')
# pcaTestFeaturePath = os.path.join(root, 'pcaTestFeature.pkl')
# jsonPath = os.path.join(root, "colNames.json")


tagtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_tag.csv'
behtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_beh.csv'
trdtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_trd.csv'

tagtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_tag.csv'
behtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_beh.csv'
trdtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_trd.csv'

pcaDataPath = os.path.join(root, 'pcaData.npz')
processedDataPath = os.path.join(root, 'processedData.npz')
scaledDataPath = os.path.join(root, 'scaledData.npz')
reinbalancedDataPath = os.path.join(root, "reinbalancedData.npy")



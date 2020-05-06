def _reinbalance(features, target):
    inbalanced_arr = np.c_[features, target]
    print("Balancing!")
    result = SMOTE(inbalanced_arr)
    result = pd.DataFrame(result, columns=valid_features + ['flag'])
    result.to_pickle("resampled_test_feature.pkl")
    return result

def reinbalance(features, target, update=False):
    if update:
        return _reinbalance(features, target)
    if os.path.exists(reinbalanced_data_path):
        return pd.read_pickle(reinbalanced_data_path)
    else:
        return _reinbalance(features, target)

reinbalanced_data = reinbalance(train_tag[featureCols], train_tag[targetCol])

X_train_tag, X_val_tag, y_train_tag, y_val_tag = train_test_split(
    reinbalanced_data[valid_features],
    reinbalanced_data['flag'], test_size=0.05, random_state=10
)

X_train_tag, X_val_tag, y_train_tag, y_val_tag = train_test_split(
    train_tag[valid_features],
    train_tag[target], test_size=0.05, random_state=10
)

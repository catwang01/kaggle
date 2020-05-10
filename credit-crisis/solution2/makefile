clean:
	python3 extractFeatures.py --verbose=False
	python3 rebalance.py
	python3 scale.py
	python3 pca.py
train:
	python3 lgb.py
	python3 rf.py

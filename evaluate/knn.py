import os.path as osp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import sys
feature_root = sys.argv[1]
train_features = np.load(osp.join(feature_root, 'train_features.npy'))
train_labels = np.load(osp.join(feature_root, 'train_labels.npy'))
test_features = np.load(osp.join(feature_root, 'test_features.npy'))
test_labels = np.load(osp.join(feature_root, 'test_labels.npy'))

print(f"num train samples: {train_labels.shape[0]}")
print(f"num test samples: {test_labels.shape[0]}")

print(f"feature ndim: {test_features.shape[1]}")

# 创建 kNN 分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(train_features, train_labels)

# 预测标签
predicted_labels = knn.predict(test_features)

# 计算准确率
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {accuracy}")


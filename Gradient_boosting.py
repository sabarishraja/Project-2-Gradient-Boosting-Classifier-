import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                total_mse = (left_mse * np.sum(left_mask) + right_mse * np.sum(right_mask)) / len(y)

                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(value=np.mean(y))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=feature, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

class CustomGradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_odds(self, y):
        p = np.clip(np.mean(y), 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        self.initial_prediction = self._log_odds(y)
        current_pred = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - self._sigmoid(current_pred)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            update = tree.predict(X)
            current_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict_proba(self, X):
        pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

def main():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start = time.time()
    model = CustomGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    custom_time = time.time() - start
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Custom Model Accuracy: {acc:.4f}, Time: {custom_time:.2f}s")

    start = time.time()
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    skl_time = time.time() - start
    skl_pred = gb.predict(X_test)
    skl_acc = accuracy_score(y_test, skl_pred)
    print(f"Sklearn Model Accuracy: {skl_acc:.4f}, Time: {skl_time:.2f}s")

if __name__ == "__main__":
    main()

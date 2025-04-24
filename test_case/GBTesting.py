import numpy as np
import unittest
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from Gradient_boosting import CustomGradientBoostingClassifier  # Update with actual path


class TestGradientBoostingClassifier(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        }

    def _train_and_validate(self, X, y, params=None, min_accuracy=0.7, title="Test"):
        params = params or self.default_params
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        model = CustomGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_custom = accuracy_score(y_test, y_pred)

        sk_model = SklearnGBC(**params)
        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)
        acc_sklearn = accuracy_score(y_test, sk_pred)

        print(f"\n{title}")
        print(f"Custom Accuracy:  {acc_custom:.4f}")
        print(f"Sklearn Accuracy: {acc_sklearn:.4f}")
        self.assertGreaterEqual(acc_custom, min_accuracy)

        self._plot_results(X_test, y_test, y_pred, sk_pred, title)

        return acc_custom, acc_sklearn

    def _plot_results(self, X, y_true, y_custom, y_sklearn, title):
        if X.shape[1] > 2:
            return  

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", s=10)
        axs[0].set_title("Ground Truth")

        axs[1].scatter(X[:, 0], X[:, 1], c=y_custom, cmap="coolwarm", s=10)
        axs[1].set_title("Custom Model")

        axs[2].scatter(X[:, 0], X[:, 1], c=y_sklearn, cmap="coolwarm", s=10)
        axs[2].set_title("Sklearn Model")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def test_basic_functionality(self):
        X, y = make_classification(n_samples=1000, n_features=2,
                                   n_informative=2, n_redundant=0, random_state=self.seed)
        self._train_and_validate(X, y, title="Basic Functionality")

    def test_high_learning_rate(self):
        X, y = make_classification(n_samples=800, n_features=2,
                                   n_redundant=0, random_state=self.seed)
        params = {'learning_rate': 0.9, 'n_estimators': 50, 'max_depth': 3}
        self._train_and_validate(X, y, params, min_accuracy=0.65, title="High Learning Rate")

    def test_low_learning_rate(self):
        X, y = make_classification(
            n_samples=1200, n_features=2, n_informative=2,
            n_redundant=0, n_repeated=0, n_clusters_per_class=1,
            random_state=self.seed
        )
        params = {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 3}
        self._train_and_validate(X, y, params, title="Low Learning Rate")

    def test_shallow_trees(self):
        X, y = make_classification(n_samples=1000, n_features=2,
                                   n_informative=2, n_redundant=0, random_state=self.seed)
        params = {'max_depth': 1, 'n_estimators': 150, 'learning_rate': 0.1}
        self._train_and_validate(X, y, params, title="Shallow Trees")

    def test_noisy_data(self):
        X, y = make_classification(
            n_samples=1000, n_features=2, n_informative=2,
            n_redundant=0, n_repeated=0, flip_y=0.3,
            random_state=self.seed
        )
        self._train_and_validate(X, y, min_accuracy=0.6, title="Noisy Data")

    def test_imbalanced_classes(self):
        X, y = make_classification(
            n_samples=1000, n_features=2, n_informative=2,
            n_redundant=0, n_repeated=0, weights=[0.9, 0.1],
            flip_y=0.01, random_state=self.seed
        )
        self._train_and_validate(X, y, min_accuracy=0.7, title="Imbalanced Classes")

    def test_non_linear_boundary(self):
        X, y = make_blobs(n_samples=1000, centers=2,
                          cluster_std=3.0, random_state=self.seed)
        self._train_and_validate(X, y, title="Non-linear Boundary")


if __name__ == '__main__':
    unittest.main()
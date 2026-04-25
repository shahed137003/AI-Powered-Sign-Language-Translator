from sklearn.decomposition import PCA

class FeaturePCA:
    def __init__(self, variance=0.98):
        self.pca = PCA(n_components=variance)

    def fit(self, X):
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)
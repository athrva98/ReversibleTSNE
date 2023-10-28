import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

SUPPORTED_MODELS = {RandomForestRegressor, KNeighborsRegressor, MultiOutputRegressor}
warnings.simplefilter("default") # for debugging : use "error"

def _train_model(X, y, model, model_args):
    # Check if model_args provides "args" and "kwargs", if not, use default empty values
    args = model_args.get("args", ())
    kwargs = model_args.get("kwargs", {})

    # Instantiate the model using provided args and kwargs
    regressor = model(*args, **kwargs)

    regressor.fit(X, y) # fit the model
    return regressor

class ReversibleTSNE(TSNE):
    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
                 n_iter=9000, n_iter_without_progress=300, min_grad_norm=1e-7, metric="euclidean",
                 init="pca", verbose=0, random_state=None, method="barnes_hut", angle=0.5, n_jobs=-1,
                 explained_variance_for_dimensionality_reduction=95,
                 threshold_dimensions=20, approximate=False,
                 model_args={"model": RandomForestRegressor, # model
                             "args": [1000],  # n_estimators / n_neighbors
                             "kwargs": {"bootstrap" : False,
                                        "criterion" : "poisson",
                                        "n_jobs" : -1}
                        }):
        # Initialize parent TSNE with its parameters
        super().__init__(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration,
                         learning_rate=learning_rate, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                         min_grad_norm=min_grad_norm, metric=metric, init=init, verbose=verbose, random_state=random_state,
                         method=method, angle=angle, n_jobs=n_jobs)

        if model_args["model"] not in SUPPORTED_MODELS:
            raise ValueError(f"Provided model {model_args['model']} is not supported. Supported models are: {', '.join([m.__name__ for m in SUPPORTED_MODELS])}")

        # Custom parameters for ReversibleTSNE
        self.explained_variance_for_dimensionality_reduction = explained_variance_for_dimensionality_reduction / 100.0
        self.threshold_dimensions = threshold_dimensions
        self.approximate = approximate
        self.model_args = model_args
        self._is_fitted = False
        self.scaler = MinMaxScaler()

    def _determine_pca_components(self, data):
        pca = PCA().fit(data)
        explained_variances = pca.explained_variance_ratio_.cumsum()
        n_components = np.argmax(explained_variances >= self.explained_variance_for_dimensionality_reduction) + 1
        return n_components

    def fit(self, X, y=None):
        # Call parent's fit method
        super().fit(X)
        self._is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        transformed_data = super().fit_transform(X)
        self._is_fitted = True
        return transformed_data

    def inverse_transform(self, embedding, original_data):
        if not self._is_fitted:
            raise ValueError("You should fit the model first before calling inverse_transform.")

        self.linearEmbeddingGenerator = None
        self.X = original_data
        self.y = self.fit_transform(original_data)  # this is the t-SNE embedding
        self.scaler.fit(self.y)
        if self.approximate:
            self.n_components_reconstruction = self._determine_pca_components(self.X)
            if self.n_components_reconstruction > self.threshold_dimensions:
                warnings.warn(f"The determined n_components is {self.n_components_reconstruction}, which is large. Consider reducing 'explained_variance_for_dimensionality_reduction' for efficiency.")
            self.linearEmbeddingGenerator = PCA(n_components=self.n_components_reconstruction)
            # apply linear dimensionality reduction
            self.X = self.linearEmbeddingGenerator.fit_transform(self.X)

        # learn an inverse mapping from the output to the input.
        self.inverseMappingGenerator = _train_model(np.exp(13.7 * self.scaler.transform(self.y)), np.exp(self.X), self.model_args["model"], self.model_args)
        del self.X  # free up this memory
        del self.y
        levelOnePrediction = np.log(self.inverseMappingGenerator.predict(np.exp(13.7 * self.scaler.transform(embedding.reshape(-1, self.n_components)))))
        return (self.linearEmbeddingGenerator.inverse_transform(levelOnePrediction.reshape(-1, self.n_components_reconstruction))) \
                if (self.linearEmbeddingGenerator is not None) else (levelOnePrediction.reshape(-1, original_data.shape[-1]))

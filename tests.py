import unittest
import numpy as np
from ReversibleTSNE import ReversibleTSNE
from sklearn.svm import SVR

class TestReversibleTSNE(unittest.TestCase):
    def setUp(self):
        # Generate some dummy data for testing
        self.data = np.random.rand(1000, 50)
        self.tsne = ReversibleTSNE(perplexity=9, n_components=2, approximate=False, method="barnes_hut")

    def test_determine_pca_components(self):
        n_components = self.tsne._determine_pca_components(self.data)
        self.assertGreaterEqual(n_components, 1)
        self.assertLessEqual(n_components, 50)

        # Test with different "explained_variance_for_dimensionality_reduction" values
        tsne_80 = ReversibleTSNE(explained_variance_for_dimensionality_reduction=80)
        n_components_80 = tsne_80._determine_pca_components(self.data)
        self.assertLessEqual(n_components_80, n_components)

    def test_transformation_and_inverse(self):
        transformed = self.tsne.fit_transform(self.data)

        # Not using approximation
        reconstructed = self.tsne.inverse_transform(transformed, self.data)
        self.assertEqual(reconstructed.shape, self.data.shape)
        self.assertFalse(np.array_equal(reconstructed, self.data))

        # Using approximation
        tsne_approx = ReversibleTSNE(perplexity=9, n_components=2, approximate=True)
        transformed_approx = tsne_approx.fit_transform(self.data)
        reconstructed_approx = tsne_approx.inverse_transform(transformed_approx, self.data)
        self.assertEqual(reconstructed_approx.shape, self.data.shape)
        self.assertFalse(np.array_equal(reconstructed_approx, self.data))

    def test_unsupported_models(self):
        with self.assertRaises(ValueError):
            ReversibleTSNE(model_args={"model": SVR})

if __name__ == "__main__":
    unittest.main(verbosity=2)

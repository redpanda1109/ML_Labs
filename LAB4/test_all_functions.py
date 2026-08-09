import unittest
import numpy as np
import pandas as pd

from l4q2_3 import data_clean, ai_label, ai_onehot
from l4q4_5_6 import minkowski_distance, apply_minkowski
from l4q7 import dot_product, euclidean_norm
from l4q8_9 import self_statistics
from l4q10 import histogram
from l4q11 import distance, kmeans


class TestEncoding(unittest.TestCase):
#label encoding
    # Normal Case
    def test_label_encoding_normal(self):
        df = pd.DataFrame({"Education": [ "Graduation", "PhD", "Master"]})
        ordinal = ["Education"]
        result = ai_label(df.copy(), ordinal)
        self.assertTrue(
            pd.api.types.is_numeric_dtype(result["Education"])
        )
        self.assertEqual(
            result["Education"].nunique(),
            3
        )

    # Boundary Case
    def test_label_encoding_single_category(self):
        df = pd.DataFrame({"Education": [ "Graduation", "Graduation", "Graduation"]})
        result = ai_label(
            df.copy(),
            ["Education"]
        )
        self.assertEqual(
            result["Education"].nunique(),
            1
        )
        self.assertTrue(
            (result["Education"] == 0).all()
        )

    # Edge Case
    def test_label_encoding_empty_dataframe(self):
        df = pd.DataFrame({"Education": pd.Series([], dtype="object")})
        result = ai_label(
            df.copy(),
            ["Education"]
        )
        self.assertEqual(
            len(result),
            0
        )

# One hot encoding
    # Normal Case
    def test_onehot_encoding_normal(self):
        df = pd.DataFrame({
            "City": [
                "Bangalore",
                "Hyderabad",
                "Chennai"
            ]
        })
        nominal = ["City"]
        result = ai_onehot(
            df.copy(),
            nominal
        )
        self.assertNotIn(
            "City",
            result.columns
        )
        self.assertEqual(
            len(result.columns),
            3
        )

    # Boundary Case
    def test_onehot_encoding_single_category(self):
        df = pd.DataFrame({
            "City": [
                "Bangalore",
                "Bangalore",
                "Bangalore"]
        })
        result = ai_onehot(
            df.copy(),
            ["City"]
        )
        self.assertEqual(
            len(result.columns),
            1
        )
        self.assertTrue(
            (result.iloc[:, 0] == 1).all()
        )

    # Edge Case
    def test_onehot_encoding_empty_dataframe(self):
        df = pd.DataFrame({
            "City": pd.Series([], dtype="object")})
        result = ai_onehot(
            df.copy(),
            ["City"]
        )
        self.assertEqual(
            len(result),
            0
        )


class TestMinkowski(unittest.TestCase):
# p=1
    def test_p_one(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = minkowski_distance(a, b, 1)
        self.assertAlmostEqual(
            result,
            9
        )

# p=2 
    def test_p_two(self):
        a = [0, 0]
        b = [3, 4]
        result = minkowski_distance(a, b, 2)
        self.assertAlmostEqual(
            result,
            5
        )

# p=10
    def test_p_ten(self):
        a = [1, 2]
        b = [2, 3]
        result = minkowski_distance(a, b, 10)
        expected = 2 ** (1 / 10)
        self.assertAlmostEqual(
            result,
            expected
        )

# Identical vectors
    def test_identical_vectors(self):
        a = [10, 20, 30]
        b = [10, 20, 30]
        result = minkowski_distance(a, b, 2)
        self.assertAlmostEqual(
            result,
            0)


class TestVectorOperations(unittest.TestCase):
# Dot product
    # Normal Case
    def test_dot_product_normal(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = dot_product(a, b)
        self.assertEqual(
            result,
            32
        )

    # Boundary Case - Single element
    def test_dot_product_single_element(self):
        a = [5]
        b = [4]
        result = dot_product(a, b)
        self.assertEqual(
            result,
            20
        )

    # Edge Case - Zero vector
    def test_dot_product_zero_vector(self):
        a = [0, 0, 0]
        b = [1, 2, 3]
        result = dot_product(a, b)
        self.assertEqual(
            result,
            0
        )
# Euclidean norm
    # Normal Case
    def test_norm_normal(self):
        a = [3, 4]
        b = [5, 12]
        norm_a, norm_b = euclidean_norm(a, b)
        self.assertEqual(norm_a, 5)
        self.assertEqual(norm_b, 13)

    # Boundary Case - Single element
    def test_norm_single_element(self):
        a = [7]
        b = [10]
        norm_a, norm_b = euclidean_norm(a, b)
        self.assertEqual(norm_a, 7)
        self.assertEqual(norm_b, 10)

    # Edge Case - Zero vectors
    def test_norm_zero_vectors(self):
        a = [0, 0, 0]
        b = [0, 0, 0]
        norm_a, norm_b = euclidean_norm(a, b)
        self.assertEqual(norm_a, 0)
        self.assertEqual(norm_b, 0)


class TestStatistics(unittest.TestCase):
    # Normal Case
    def test_statistics_normal(self):
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        mean, variance, std = self_statistics(X)
        expected_mean = np.array([3, 4])
        expected_variance = np.array([
            8 / 3,
            8 / 3
        ])
        expected_std = np.sqrt(
            expected_variance
        )
        np.testing.assert_array_almost_equal(
            mean,
            expected_mean
        )
        np.testing.assert_array_almost_equal(
            variance,
            expected_variance
        )
        np.testing.assert_array_almost_equal(
            std,
            expected_std
        )

    # Boundary Case - Single row
    def test_statistics_single_row(self):
        X = np.array([
            [5, 10, 15]
        ])
        mean, variance, std = self_statistics(X)
        np.testing.assert_array_equal(
            mean,
            [5, 10, 15]
        )
        np.testing.assert_array_equal(
            variance,
            [0, 0, 0]
        )
        np.testing.assert_array_equal(
            std,
            [0, 0, 0]
        )

    # Edge Case - Constant values
    def test_statistics_constant_values(self):
        X = np.array([
            [5, 5],
            [5, 5],
            [5, 5]
        ])
        mean, variance, std = self_statistics(X)
        np.testing.assert_array_equal(
            variance,
            [0, 0]
        )
        np.testing.assert_array_equal(
            std,
            [0, 0]
        )


class TestHistogram(unittest.TestCase):
    # Normal Case
    def test_histogram_normal(self):
        df = pd.DataFrame({"Income": [10000,25000,50000,75000,100000]})
        try:
            histogram(df)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)

    # Boundary Case - Single value
    def test_histogram_single_value(self):
        df = pd.DataFrame({"Income": [50000]})
        try:
            histogram(df)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)

    # Edge Case - Missing value
    def test_histogram_missing_values(self):
        df = pd.DataFrame({"Income": [25000,np.nan,75000]})
        try:
            histogram(df)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)


class TestKMeans(unittest.TestCase):
#Distance Function
    # Normal Case
    def test_distance_normal(self):
        point = [0, 0]
        centroid = [3, 4]
        result = distance(
            point,
            centroid
        )
        self.assertEqual(
            result,
            5
        )

    # Boundary Case - Same point
    def test_distance_same_point(self):
        point = [5, 5]
        centroid = [5, 5]
        result = distance(
            point,
            centroid
        )
        self.assertEqual(
            result,
            0
        )

    # Edge Case - Zero vectors
    def test_distance_zero_vectors(self):
        point = [0, 0, 0]
        centroid = [0, 0, 0]
        result = distance(
            point,
            centroid
        )
        self.assertEqual(
            result,
            0
        )

#KMean Algorithm
    # Normal Case
    def test_kmeans_normal(self):
        data = [
            [1, 1],
            [1, 2],
            [2, 1],
            [10, 10],
            [10, 11],
            [11, 10]]
        k = 2
        clusters, centroids = kmeans(
            data,
            k
        )
        self.assertEqual(
            len(clusters),
            2
        )
        self.assertEqual(
            len(centroids),
            2
        )
        total_points = sum(
            len(cluster)
            for cluster in clusters
        )
        self.assertEqual(
            total_points,
            len(data)
        )

    # Boundary Case - Minimum practical K
    def test_kmeans_k_three(self):
        data = [[1, 1],
            [2, 2],
            [3, 3],
            [10, 10],
            [11, 11]]
        k = 3
        clusters, centroids = kmeans(
            data,
            k
        )
        self.assertEqual(
            len(clusters),
            3
        )
        self.assertEqual(
            len(centroids),
            3
        )

    # Edge Case - Identical points
    def test_kmeans_identical_points(self):
        data = [[5, 5],
            [5, 5],
            [5, 5],
            [5, 5]]
        k = 2
        clusters, centroids = kmeans(
            data,
            k
        )
        self.assertEqual(
            len(clusters),
            2
        )
        total_points = sum(
            len(cluster)
            for cluster in clusters
        )
        self.assertEqual(
            total_points,
            4
        )


if __name__ == "__main__":
    unittest.main()
    
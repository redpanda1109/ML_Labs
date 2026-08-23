# AI Tool Used: ChatGPT
# Lab 6 - Unit test cases for Lab 5 and Lab 6 k-NN implementations
# self knn, self weighted knn, lab5 package knn, ai knn, ai weighted knn


import unittest
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from lab5_knn import (quick_sort,k_neighbour,k_classes,winning_class)
from lab5_wknn import (quick_sort as wknn_quick_sort,k_neighbour as wknn_k_neighbour,wk_classes,winning_class_weighted)
from ai_knn import (ai_sort_distances,ai_knn_neighbors,ai_majority_knn,ai_knn_predict)
from ai_wknn import (ai_wknn_sort_distances,ai_wknn_neighbors,ai_class_weights,ai_weighted_majority,ai_wknn_predict)
from ai_fit_predict_score import (ai_fit,ai_predict,ai_wknn_predict as model_wknn_predict,ai_score)

# 1. TEST LAB 5 NORMAL K-NN
class TestLab5KNN(unittest.TestCase):
    def setUp(self):
        self.distances = [(5.0, 0),(2.0, 1),(8.0, 2),(1.0, 3)]
        self.y_train = pd.Series(["CONDITION","CONDITION","NO CONDITION","NO CONDITION"])

    # Quick Sort
    # Normal
    def test_quick_sort_normal(self):
        result = quick_sort(self.distances)
        expected = [(1.0, 3),(2.0, 1),(5.0, 0),(8.0, 2)]
        self.assertEqual(result,expected)
    # Boundary
    def test_quick_sort_boundary_single_item(self):
        distances = [(2.0, 0)]
        result = quick_sort(distances)
        self.assertEqual(result,distances)
    #Edge 
    def test_quick_sort_edge_equal_distances(self):
        distances = [(2.0, 5),(2.0, 1),(1.0, 3),(2.0, 2)]
        result = quick_sort(distances)
        expected = [(1.0, 3),(2.0, 1),(2.0, 2),(2.0, 5)]
        self.assertEqual(result,expected)


    # k-Neighbour
    # Normal
    def test_k_neighbour_normal(self):
        sorted_distances = [(1.0, 3),(2.0, 1),(5.0, 0),(8.0, 2)]
        result = k_neighbour(sorted_distances,2)
        expected = [(1.0, 3),(2.0, 1)]
        self.assertEqual(result,expected)
    # Boundary
    def test_k_neighbour_boundary_k_one(self):
        sorted_distances = [(1.0, 3),(2.0, 1),(5.0, 0)]
        result = k_neighbour(sorted_distances,1)
        self.assertEqual(result,[(1.0, 3)])
    # Edge
    def test_k_neighbour_boundary_all_samples(self):
        sorted_distances = [(1.0, 3),(2.0, 1),(5.0, 0)]
        result = k_neighbour(sorted_distances,3)
        self.assertEqual(result,sorted_distances)


    # Class extraction
    #Normal
    def test_k_classes_normal(self):
        neighbours = [(1.0, 0),(2.0, 1),(3.0, 2)]
        result = k_classes(neighbours,self.y_train)
        expected = ["CONDITION","CONDITION","NO CONDITION"]
        self.assertEqual(result,expected)
    # Boundary
    def test_k_classes_boundary_single_neighbour(self):
        neighbours = [(1.0, 0)]
        result = k_classes(neighbours,self.y_train)
        self.assertEqual(result,["CONDITION"])
    #Edge
    def test_k_classes_edge_empty_neighbours(self):
        result = k_classes([],self.y_train)
        self.assertEqual(result,[])


    # Majority voting
    # Normal
    def test_winning_class_normal_condition_majority(self):
        class_type = ["CONDITION","CONDITION","NO CONDITION"]
        result = winning_class(class_type)
        self.assertEqual(result,"CONDITION")

    def test_winning_class_normal_no_condition_majority(self):
        class_type = ["NO CONDITION","CONDITION","NO CONDITION"]
        result = winning_class(class_type)
        self.assertEqual(result,"NO CONDITION")
    # Boundary
    def test_winning_class_boundary_single_class(self):
        result = winning_class(["CONDITION"])
        self.assertEqual(result,"CONDITION")
    # Edge
    def test_winning_class_edge_tie(self):
        class_type = ["CONDITION","NO CONDITION"]
        result = winning_class(class_type) # Lab 5 tie-breaking rule: choose the first class in the neighbour order.
        self.assertEqual(result,"CONDITION")



# 2. TEST LAB 5 WEIGHTED K-NN
class TestLab5WKNN(unittest.TestCase):
    def setUp(self):
        self.y_train = pd.Series(["CONDITION","CONDITION","NO CONDITION","NO CONDITION"])
        self.distances = [(1.0, 0),(2.0, 1),(4.0, 2),(5.0, 3)]

    # Quick Sort
    # Normal
    def test_quick_sort_normal(self):
        distances = [(5.0, 0),(2.0, 1),(8.0, 2),(1.0, 3)]
        result = wknn_quick_sort(distances)
        expected = [(1.0, 3),(2.0, 1),(5.0, 0),(8.0, 2)]
        self.assertEqual(result,expected)
    # Boundary
    def test_quick_sort_boundary_single_item(self):
        distances = [(2.0, 0)]
        result = wknn_quick_sort(distances)
        self.assertEqual(result,distances)
    # Edge
    def test_quick_sort_edge_equal_distances(self):
        distances = [(2.0, 5),(2.0, 1),(1.0, 3),(2.0, 2)]
        result = wknn_quick_sort(distances)
        expected = [(1.0, 3),(2.0, 1),(2.0, 2),(2.0, 5)]
        self.assertEqual(result,expected)


    # k-Neighbour
    # Normal
    def test_k_neighbour_normal(self):
        result = wknn_k_neighbour(self.distances,2)
        expected = [(1.0, 0),(2.0, 1)]
        self.assertEqual(result,expected)
    # Boundary
    def test_k_neighbour_boundary_k_one(self):
        result = wknn_k_neighbour(self.distances,1)
        self.assertEqual(result,[(1.0, 0)])
    # Edge
    def test_k_neighbour_edge_k_zero(self):
        result = wknn_k_neighbour(self.distances,0)
        self.assertEqual(result,[])


    # Weighted class extraction
    # Normal
    def test_wk_classes_normal(self):
        neighbours = [(1.0, 0),(2.0, 1),(4.0, 2)]
        result = wk_classes(neighbours,self.y_train)
        expected = ["CONDITION","CONDITION","NO CONDITION"]
        self.assertEqual(result,expected)
    # Boundary
    def test_wk_classes_boundary_single_neighbour(self):
        neighbours = [(1.0, 0)]
        result = wk_classes(neighbours,self.y_train)
        self.assertEqual(result,["CONDITION"])
    #Edge
    def test_wk_classes_edge_empty_neighbours(self):
        result = wk_classes([],self.y_train)
        self.assertEqual(result,[])


    # Weighted majority
    def test_weighted_majority_normal_condition(self):
        neighbours = [(1.0, 0),(2.0, 1),(4.0, 2)]
        class_type = ["CONDITION","CONDITION","NO CONDITION"]
        result = winning_class_weighted(class_type,neighbours)
        self.assertEqual(result,"CONDITION")

    def test_weighted_majority_normal_no_condition(self):
        neighbours = [(1.0, 0),(2.0, 1),(3.0, 2)]
        class_type = ["CONDITION","NO CONDITION","NO CONDITION"]
        result = winning_class_weighted(class_type,neighbours)
        self.assertEqual(result,"NO CONDITION")

    def test_weighted_majority_boundary_single_neighbour(self):
        neighbours = [(2.0, 0)]
        class_type = ["CONDITION"]
        result = winning_class_weighted(class_type,neighbours)
        self.assertEqual(result,"CONDITION")

    def test_weighted_zero_distance_edge_case(self):
        neighbours = [(0.0, 0),(2.0, 1)]
        class_type = ["CONDITION","NO CONDITION"] # The original Lab 5 implementation uses 1 / distance directly. Therefore an exact distance of zero raises ZeroDivisionError.
        with self.assertRaises(ZeroDivisionError):
            winning_class_weighted(class_type,neighbours)



# 3. TEST LAB 5 PACKAGE K-NN
class TestLab5PackageKNN(unittest.TestCase):
    def setUp(self):
        self.X_train = [[0, 0],[0, 1],[10, 10],[10, 11]]
        self.y_train = ["CONDITION","CONDITION","NO CONDITION","NO CONDITION"]
        self.X_test = [[0, 0.5],[10, 10.5]]
        self.y_test = ["CONDITION","NO CONDITION"]

    def test_package_knn_prediction_normal(self):
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(self.X_train,self.y_train)
        predictions = model.predict(self.X_test)
        expected = ["CONDITION","NO CONDITION"]
        self.assertEqual(list(predictions),expected)

    def test_package_knn_prediction_boundary_k_all_samples(self):
        model = KNeighborsClassifier(n_neighbors=4)
        model.fit(self.X_train,self.y_train)
        predictions = model.predict([[0, 0.5]])
        self.assertEqual(len(predictions),1)

    def test_package_knn_prediction_edge_invalid_k(self):
        with self.assertRaises(ValueError):
            KNeighborsClassifier(n_neighbors=0)

    def test_package_knn_score_normal(self):
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(self.X_train,self.y_train)
        score = model.score(self.X_test,self.y_test)
        self.assertEqual(score,1.0)

    def test_package_knn_score_boundary_zero_accuracy(self):
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(self.X_train,self.y_train)
        wrong_labels = ["NO CONDITION","CONDITION"]
        score = model.score(self.X_test,wrong_labels)
        self.assertEqual(score,0.0)


# 4. TEST AI K-NN
class TestAIKNN(unittest.TestCase):
    def setUp(self):
        self.distances = [(5.0, 0),(2.0, 1),(8.0, 2),(1.0, 3)]
        self.X_train = [[0, 0],[0, 1],[5, 5],[5, 6]]
        self.y_train = ["CONDITION","CONDITION","NO CONDITION","NO CONDITION"]


    # Quick Sort
    def test_sort_normal(self):
        result = ai_sort_distances(self.distances)
        expected = [(1.0, 3),(2.0, 1),(5.0, 0),(8.0, 2)]
        self.assertEqual(result,expected)

    def test_sort_boundary_single_item(self):
        distances = [(2.0, 0)]
        result = ai_sort_distances(distances)
        self.assertEqual(result,distances)

    def test_sort_edge_equal_distances(self):
        distances = [(2.0, 5),(2.0, 1),(1.0, 3),(2.0, 2)]
        result = ai_sort_distances(distances)
        expected = [(1.0, 3),(2.0, 1),(2.0, 2),(2.0, 5)]
        self.assertEqual(result,expected)

    def test_sort_edge_empty_list(self):
        result = ai_sort_distances([])
        self.assertEqual(result,[])


    # k-Neighbours
    def test_neighbors_normal(self):
        result = ai_knn_neighbors(
            [(1.0, 3),(2.0, 1),(5.0, 0)],2)
        expected = [(1.0, 3),(2.0, 1)]
        self.assertEqual(result,expected)

    def test_neighbors_boundary_k_one(self):
        result = ai_knn_neighbors(self.distances,1)
        self.assertEqual(result,[(5.0, 0)])

    def test_neighbors_boundary_k_all(self):
        result = ai_knn_neighbors(self.distances,4)
        self.assertEqual(result,self.distances)

    def test_neighbors_edge_k_zero(self):
        with self.assertRaises(ValueError):
            ai_knn_neighbors(self.distances,0)

    def test_neighbors_edge_k_too_large(self):
        with self.assertRaises(ValueError):
            ai_knn_neighbors(self.distances,10)


    # Majority voting
    def test_majority_normal_condition(self):
        neighbors = [(1.0, 0),(2.0, 1),(3.0, 2)]
        y_train = ["CONDITION","CONDITION","NO CONDITION"]
        result = ai_majority_knn(neighbors,y_train)
        self.assertEqual(result,"CONDITION")

    def test_majority_normal_no_condition(self):
        neighbors = [(1.0, 0),(2.0, 1),(3.0, 2)]
        y_train = ["NO CONDITION","NO CONDITION","CONDITION"]
        result = ai_majority_knn(neighbors,y_train)
        self.assertEqual(result,"NO CONDITION")

    def test_majority_boundary_single_neighbour(self):
        result = ai_majority_knn([(1.0, 0)],["CONDITION"])
        self.assertEqual(result,"CONDITION")

    def test_majority_edge_tie_closest_neighbor(self):
        neighbors = [(0.5, 0),(1.0, 1)]
        y_train = ["CONDITION","NO CONDITION"]
        result = ai_majority_knn(neighbors,y_train)
        self.assertEqual(result,"CONDITION")


    # Complete AI k-NN prediction
    def test_knn_prediction_normal(self):
        X_test = [[0, 0.2],[5, 5.2]]
        predictions, counts = ai_knn_predict(self.X_train,self.y_train,X_test,3)
        expected = ["CONDITION","NO CONDITION"]
        self.assertEqual(predictions,expected)
        self.assertEqual(counts["CONDITION"],1)
        self.assertEqual(counts["NO CONDITION"],1)

    def test_knn_prediction_boundary_single_test_vector(self):
        predictions, counts = ai_knn_predict(self.X_train,self.y_train,[[0, 0.2]],1)
        self.assertEqual(len(predictions),1)
        self.assertEqual(counts["CONDITION"],1)

    def test_knn_prediction_edge_invalid_k(self):
        with self.assertRaises(ValueError):
            ai_knn_predict(self.X_train,self.y_train,[[0, 0.2]],0)



# 5. TEST AI WEIGHTED K-NN
class TestAIWKNN(unittest.TestCase):
    def setUp(self):
        self.distances = [(5.0, 0),(2.0, 1),(8.0, 2),(1.0, 3)]
        self.X_train = [[0, 0],[0, 1],[5, 5],[5, 6]]
        self.y_train = ["CONDITION","CONDITION","NO CONDITION","NO CONDITION"]


    # Quick Sort
    def test_sort_normal(self):
        result = ai_wknn_sort_distances(self.distances)
        expected = [(1.0, 3),(2.0, 1),(5.0, 0),(8.0, 2)]
        self.assertEqual(result,expected)

    def test_sort_boundary_single_item(self):
        distances = [(2.0, 0)]
        result = ai_wknn_sort_distances(distances)
        self.assertEqual(result,distances)

    def test_sort_edge_equal_distances(self):
        distances = [(2.0, 5),(2.0, 1),(1.0, 3),(2.0, 2)]
        result = ai_wknn_sort_distances(distances)
        expected = [(1.0, 3),(2.0, 1),(2.0, 2),(2.0, 5)]
        self.assertEqual(result,expected)


    # k-Neighbours
    def test_neighbors_normal(self):
        result = ai_wknn_neighbors(self.distances,2)
        expected = [(5.0, 0),(2.0, 1)]
        self.assertEqual(result,expected)

    def test_neighbors_boundary_k_one(self):
        result = ai_wknn_neighbors(self.distances,1)
        self.assertEqual(result,[(5.0, 0)])

    def test_neighbors_boundary_k_all(self):
        result = ai_wknn_neighbors(self.distances,4)
        self.assertEqual(result,self.distances)

    def test_neighbors_edge_invalid_k(self):
        with self.assertRaises(ValueError):
            ai_wknn_neighbors(self.distances,0)


    # Class weights
    def test_class_weights_normal(self):
        neighbors = [(1.0, 0),(2.0, 1),(4.0, 2)]
        y_train = ["CONDITION","CONDITION","NO CONDITION"]
        result = ai_class_weights(neighbors,y_train)
        self.assertAlmostEqual(result["CONDITION"],1.5)
        self.assertAlmostEqual(result["NO CONDITION"],0.25)

    def test_class_weights_boundary_single_neighbour(self):
        result = ai_class_weights([(2.0, 0)],["CONDITION"])
        self.assertAlmostEqual(result["CONDITION"],0.5)

    def test_class_weights_edge_zero_distance(self):
        neighbors = [(0.0, 0),(2.0, 1)]
        y_train = ["CONDITION","NO CONDITION"]
        result = ai_class_weights(neighbors,y_train)
        self.assertEqual(result["CONDITION"],float("inf"))
        self.assertAlmostEqual(result["NO CONDITION"],0.5)


    # Weighted majority
    def test_weighted_majority_normal_condition(self):
        neighbors = [(1.0, 0),(2.0, 1),(4.0, 2)]
        y_train = ["CONDITION","CONDITION","NO CONDITION"]
        class_weights = {"CONDITION": 1.5,"NO CONDITION": 0.25}
        result = ai_weighted_majority(neighbors,y_train,class_weights)
        self.assertEqual(result,"CONDITION")

    def test_weighted_majority_normal_no_condition(self):
        neighbors = [(1.0, 0),(2.0, 1),(3.0, 2)]
        y_train = ["CONDITION","NO CONDITION","NO CONDITION"]
        class_weights = {"CONDITION": 1.0,"NO CONDITION": 1.1666666667}
        result = ai_weighted_majority(neighbors,y_train,class_weights)
        self.assertEqual(result,"NO CONDITION")

    def test_weighted_majority_boundary_single_neighbour(self):
        result = ai_weighted_majority([(2.0, 0)],["CONDITION"],{"CONDITION": 0.5})
        self.assertEqual(result,"CONDITION")

    def test_weighted_majority_edge_tie(self):
        neighbors = [(0.5, 0),(1.0, 1)]
        y_train = ["CONDITION","NO CONDITION"]
        class_weights = {"CONDITION": 2.0,"NO CONDITION": 2.0}
        result = ai_weighted_majority(neighbors,y_train,class_weights)
        self.assertEqual(result,"CONDITION")


    # Complete weighted k-NN prediction
    def test_wknn_prediction_normal(self):
        X_test = [[0, 0.2],[5, 5.2]]
        predictions = ai_wknn_predict(self.X_train,self.y_train,X_test,3)
        expected = ["CONDITION","NO CONDITION"]
        self.assertEqual(predictions,expected)

    def test_wknn_prediction_boundary_single_test_vector(self):
        predictions = ai_wknn_predict(self.X_train,self.y_train,[[0, 0.2]],1)
        self.assertEqual(len(predictions),1)
        self.assertEqual(predictions[0],"CONDITION")

    def test_wknn_prediction_edge_zero_distance(self):
        predictions = ai_wknn_predict(self.X_train,self.y_train,[[0, 0]],3)
        self.assertEqual(predictions[0],"CONDITION")



# 6. TEST AI FIT, PREDICT AND SCORE
class TestAIFitPredictScore(unittest.TestCase):
    def setUp(self):
        self.X_train = [[0, 0],[0, 1],[5, 5],[5, 6]]
        self.y_train = ["CONDITION","CONDITION","NO CONDITION","NO CONDITION"]
        self.X_test = [[0, 0.2],[5, 5.2]]
        self.y_test = ["CONDITION","NO CONDITION"]

    # ai_fit
    def test_ai_fit_normal(self):
        training_data, training_labels = ai_fit(self.X_train,self.y_train)
        self.assertEqual(training_data,self.X_train)
        self.assertEqual(training_labels,self.y_train)

    def test_ai_fit_boundary_single_sample(self):
        X_train = [[0, 0]]
        y_train = ["CONDITION"]
        training_data, training_labels = ai_fit(X_train,y_train)
        self.assertEqual(training_data,X_train)
        self.assertEqual(training_labels,y_train)

    def test_ai_fit_edge_length_mismatch(self):
        with self.assertRaises(ValueError):
            ai_fit([[0, 0], [1, 1]],["CONDITION"])


    # ai_predict
    def test_ai_predict_normal(self):
        training_data, training_labels = ai_fit(self.X_train,self.y_train)
        predictions = ai_predict(self.X_test,training_data,training_labels,3)
        expected = ["CONDITION","NO CONDITION"]
        self.assertEqual(predictions,expected)

    def test_ai_predict_boundary_single_test_vector(self):
        training_data, training_labels = ai_fit(self.X_train,self.y_train)
        predictions = ai_predict([[0, 0.2]],training_data,training_labels,1)
        self.assertEqual(predictions,["CONDITION"])

    def test_ai_predict_edge_invalid_k(self):
        training_data, training_labels = ai_fit(self.X_train,self.y_train)
        with self.assertRaises(ValueError):
            ai_predict(self.X_test,training_data,training_labels,0)


    # ai_wknn_predict
    def test_ai_wknn_predict_normal(self):
        training_data, training_labels = ai_fit(self.X_train,self.y_train)
        predictions = model_wknn_predict(self.X_test,training_data,training_labels,3)
        expected = ["CONDITION","NO CONDITION"]
        self.assertEqual(predictions,expected)

    def test_ai_wknn_predict_boundary_single_test_vector(self):
        training_data, training_labels = ai_fit(self.X_train,self.y_train)
        predictions = model_wknn_predict([[0, 0.2]],training_data,training_labels,1)
        self.assertEqual(predictions,["CONDITION"])

    def test_ai_wknn_predict_edge_zero_distance(self):
        training_data, training_labels = ai_fit(self.X_train,self.y_train)
        predictions = model_wknn_predict([[0, 0]],training_data,training_labels,3)
        self.assertEqual(predictions,["CONDITION"])


    # ai_score
    def test_ai_score_normal_partial_accuracy(self):
        y_test = ["CONDITION","NO CONDITION","CONDITION","NO CONDITION"]
        predictions = ["CONDITION","CONDITION","CONDITION","NO CONDITION"]
        result = ai_score(y_test,predictions)
        self.assertEqual(result,0.75)

    def test_ai_score_boundary_perfect_accuracy(self):
        result = ai_score(self.y_test,self.y_test)
        self.assertEqual(result,1.0)

    def test_ai_score_boundary_zero_accuracy(self):
        wrong_predictions = ["NO CONDITION","CONDITION"]
        result = ai_score(self.y_test,wrong_predictions)
        self.assertEqual(result,0.0)

    def test_ai_score_edge_empty(self):
        with self.assertRaises(ValueError):
            ai_score([],[])

    def test_ai_score_edge_length_mismatch(self):
        with self.assertRaises(ValueError):
            ai_score(["CONDITION"],["CONDITION", "NO CONDITION"])


# ============================================================
# RUN ALL UNIT TESTS
# ============================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)

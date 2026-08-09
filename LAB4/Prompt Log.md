# Prompt Log

## Task 1 — Data Cleaning and Encoding

I have an excel sheet with some data and i have labeled two list called ordinal and nominal which contain the list of attributes in the excel dataset. Now perform data cleaning on the dataset under the function data_clean(), then define another function ai_label() to perform label encoding for the ordinal attributes and ai_onehot() to perform one-hot encoding for the nominal attributes. In the end print the resultant dataset. Do this task in a python code

---

## Task 2 — Minkowski Distance

next create another python code where we define a function minkowski_distance() to calculate the generalized minkowski distance. using this function it should calculate Manhattan or Euclidean distance and print it. For a given two numerical vectors from the dataset calculate the Minkwoski distance with p from 1 to 10 under the function apply_minkowski(), which plots a graph based on the results. Then generate another function called comparison() which compares the distance values from apply_minkowski() with the package function available as scipy.spatial.distance.minkowski()

---

## Task 3 — Dot Product and Euclidean Norm

give me another python code which the functions dot_product(), which calculates the dot product of two vectors a and b, and euclidean_norm(), which finds the length of vectors a and b

---

## Task 4 — Statistical Functions

give me another python code which first takes in all the numerical columns in the given excel dataset. Then generates a function self_statistics() that performs mean, variance and standard deviation on the dataset where the dataset represented as a matrix with columns as features. Print the output form this function. Then generate another fucntion called comparison() that compares values from self_statistics() and mean, variance and standard deviation of the dataset using inbuilt package functions available in python.

---

## Task 5 — Histogram

give me another python code with a function called histogram() where we take the Income feature from the excel dataset and observe the density pattern for that feature by plotting a histogram. Use buckets (data in ranges) for histogram generation and study.

---

## Task 6 — K-Means Implementation

help me generate a python code which implements k-means algorithm. First have a function called data_clean() to clean the data from the given excel dataset and take only numerical features. In the same function create a matrix called data where every row is appended as a list of its feature vectors. Then the main algorithm is to be done in kmeans() function with a random k value between 3-7. Also have a separate function called distance() that calculates the distance between point and cluster centroids, which is called in the main function kmeans().

---

## Task 7 — Dataset Details

"Lab Session Data.xlsx", sheet_name='marketing_campaign' my dataset is in this

---

## Task 8 — Unit Testing

ok now from all the python files generated lets create unit test cases for all the task done. Lets make a separate python file that imports all these python files and makes separate classes to test the modular functions. The unit test cases should include normal cases, boundary cases and appropriate edge cases. Maybe we can use python's built-in unittest module for this code

---

## Task 9 — Unit Test Structure

i dont want TestDataCleaaning rather we can add TestEncoding which does unit testing for label and onehot encoding for normal, boundary and edge cases. TestMinkowski for p=1, p=2, p=10 and identical vectors. TestVectorOperations has test case for dot and norm operation for normal, boundary and edge case. TestStatistics has normal, boundary and edge cases. Similarly, TestHistogram has normal, boundary and edge cases. Lastly, TestKMeans has normal, boundary and edge

---

## Task 10 — K-Means Code Comparison

i want to create a python file that compares my code for kmeans algorithm and the previously generated ai code for kmeans. Lets import the files l3q11 and l4q11 to the new python file. Then run the kmeans algorithms for k=4 and k=5. There shd be separate functions like execution time(), calculate_wcss(), silhouette_score(), cluster_size() which gives the cluster size and no.of iterations. Then finally it prints the values in the end to observe the comparison between the two codes

---

## Task 11 — Prompt Log

can you create the prompt log for this chat
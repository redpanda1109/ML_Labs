# AI tool used: ChatGPT for function creation
import numpy as np

# Function to calculate dot product
def dot_product(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b)


# Function to calculate Euclidean norm (length)
def euclidean_norm(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return norm_a, norm_b



a = [1, 2, 3]
b = [4, 5, 6]
dot = dot_product(a, b)
norm_a, norm_b = euclidean_norm(a, b)

print("Vector a:", a)
print("Vector b:", b)
print("\nDot Product of a and b:", dot)
print("Euclidean Norm of a:", norm_a)
print("Euclidean Norm of b:", norm_b)
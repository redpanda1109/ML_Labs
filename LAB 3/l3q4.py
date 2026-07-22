import math

def minkowski_dist(u,v,order):
    temp=0
    for i in range(len(u)):
        temp=math.pow(abs(u[i]-v[i]),order)
    result=math.pow(temp,(1/order))
    return result

u=[10, 11, 14, 15]
v=[16, 18, 24, 79]
print("Distance between u & v using Manhattan Dist: ", minkowski_dist(u,v,1))
print("Distance between u & v using Euclidean Dist: ", minkowski_dist(u,v,2))



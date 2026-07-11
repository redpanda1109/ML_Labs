def powered(matrix, m):
    temp=matrix
    for i in range(m-1):
        current=[[0]*n for i in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    current[i][j]+=temp[i][k]*matrix[k][j]
        temp=current
    return temp



n=int(input("Size of the matrix: "))
matrix=[]
for i in range(n):
    r=list(map(int, input("Enter row elements: ").split()))
    if len(r)!=n:
        print("Invalid input")
        exit()
    else:
        matrix.append(r)
m=int(input("Enter the power: "))
matrix=powered(matrix, m)
print("A^m matrix:")
for i in range(n):
    for j in range(n):
        print(matrix[i][j], end=" ")
    print()
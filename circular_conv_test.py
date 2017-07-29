shift = [0, 0, 0, 0, 0, 1]
vector = [1, 2, 3, 4, 5, 6]

def circular_convolution(vector, shift):
    new = [0, 0, 0, 0, 0 ,0]
    for i in range (6):
        for j in range (6):
            new[i] += vector[j] * shift[(i - j)%6]
    
    return new 

print(circular_convolution(vector, shift))
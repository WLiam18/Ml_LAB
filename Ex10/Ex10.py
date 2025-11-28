import numpy as np

# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Activation
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

# Initialize
np.random.seed(1)
W1 = np.random.randn(2, 4)
W2 = np.random.randn(4, 1)
lr = 0.5

# Train
for i in range(10000):
    # Forward
    h = sigmoid(X.dot(W1))
    o = sigmoid(h.dot(W2))
    
    # Backward
    e = y - o
    od = e * sigmoid_deriv(o)
    he = od.dot(W2.T)
    hd = he * sigmoid_deriv(h)
    
    # Update
    W2 += h.T.dot(od) * lr
    W1 += X.T.dot(hd) * lr
    
    if i % 2000 == 0:
        print(f"Loss: {np.mean(e**2):.4f}")

# Test
print("\nPredictions:", sigmoid(sigmoid(X.dot(W1)).dot(W2)).round(2))
print("Actual:", y.T)

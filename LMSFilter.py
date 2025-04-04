import numpy as np
import logging
from collections import deque

class LMSFilter:
    def __init__(self, num_taps, mu):
        self.num_taps = num_taps
        self.mu = mu
        self.weights = np.zeros(num_taps)

    def update(self, x, d):
        y = np.dot(self.weights, x)  # Predicted output
        e = d - y  # Error
        self.weights += self.mu * e * np.array(x)  # Weight update
        return y, e

# Example usage:
lms = LMSFilter(num_taps=4, mu=0.01)
x = [1, 0, 0, 0]
d = 1  # Desired output
y, e = lms.update(x, d)
print(f"Output: {y}, Error: {e}")

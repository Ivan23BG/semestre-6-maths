import numpy as np
import matplotlib.pyplot as plt
#%%
# Question 1
# implementer la dichotomie
def algo1(f, a, b, epsilon):
    x_down = min(a, b)
    x_up = max(a, b)
    while x_up - x_down > epsilon:
        x = (x_down + x_up) / 2
        if f(x) * f(x_down) <= 0:
            x_up = x
        else:
            x_down = x
    return x

def f_test(x):
    return x-1

# Test find root
print(algo1(f_test, -2, 2, 1e-5))

#%%
def f_test2(x):
    return x**2 + 1

# Test find minimum
print(algo1(f_test2, -2, 2, 1e-5))

#%%
# for instead of while version
def algo1_for(f, a, b):
    x_down = min(a, b)
    x_up = max(a, b)
    for i in range(100):
        x = (x_down + x_up) / 2
        if f(x) * f(x_down) <= 0:
            x_up = x
        else:
            x_down = x
        if x_up == x_down:
            break
    return x

# Test find root
print(algo1_for(f_test, -2, 2))

# Test find minimum
print(algo1_for(f_test2, -2, 2))


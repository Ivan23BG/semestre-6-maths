#%% Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%
# Descente de gradients et variantes

# Algorithme de la descente de gradient

# Question 1 : Coder la descente de gradient
# et garder la liste des itérés

def algo1(grad_f, x_init, gamma, maxiter, epsilon):
    x = x_init
    x_list = [x]
    for i in range(maxiter):
        x = x - gamma * grad_f(x)
        x_list.append(x)
        if np.linalg.norm(grad_f(x)) < epsilon:
            break
    return x, x_list
#%%
def f_test(x_1, x_2):
    return (x_1-1)**2 + 3*(x_2+1)**2

#%%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_test(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#%%
# Tester l'algorithme sur la fonction f_test
def grad_f_test(x):
    return np.array([2*(x[0]-1), 6*(x[1]+1)])

x_init = np.array([0, 0])
gamma = 0.01
maxiter = 1000
epsilon = 1e-5
x_min = algo1(grad_f_test, x_init, gamma, maxiter, epsilon)
print(x_min[0],"\n", x_min[1])

# %% Question 2 : Application au cas quadratique

# On pose la fonction quadratique f_(a,b)(x,y)=y^2/a+x^2/b
# tracer l'evolution graphique des f_(a,b)(x_k) dans l'algo
# avec a=b et a=1, puis 10, 50, 100
# f correspond a f_test

def f_ab(a, b, x, y):
    if a > 0 and b > 0:
        return y**2/a + x**2/b
    
def grad_f_ab(X):
    if a > 0 and b > 0:
        return np.array([2*X[0]/b, 2*X[1]/a])

#%%
gamma = 0.01
maxiter = 1000
epsilon = 1e-5
a = 1
b = 1
x_init = np.array([1, 1])
x_min = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)

#%%
# plot the evolution of f_ab(x_k)
x_list = np.array(x_min[1])
f_ab_list = f_ab(a, b, x_list[:,0], x_list[:,1])
plt.plot(f_ab_list)

#%%
# plot on same graph the evolution of f_ab(x_k) for a=1, 10, 50, 100
x_init = np.array([1, 1])
a_list = [1, 10, 50, 100]
gamma = 0.01
maxiter = 1000
epsilon = 1e-5
for a in a_list:
    b=a
    x_min = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
    x_list = np.array(x_min[1])
    f_ab_list = f_ab(a, b, x_list[:,0], x_list[:,1])
    # plot
    plt.yscale("log")
    plt.plot(f_ab_list, label="a="+str(a))
    # title
    plt.title("Evolution of f_ab(x_k) for a=1, 10, 50, 100")
    # plot with legend
    plt.legend()

print(x_min[0],"\n", x_min[1])

#%% plot the surface of f_ab

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_ab(a, b, X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#%% plot countour lines of f_ab

plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.show()

#%% plot the evolution of x_k on the countour lines of f_ab

# plot the evolution of f_ab(x_k) on different plots
for a in a_list:
    b=a
    Z = f_ab(a, b, X, Y)
    plt.contour(X, Y, Z, 20)
    plt.colorbar()
    # title
    plt.title("Evolution of f_ab(x_k) for a=1, 10, 50, 100")
    plt.plot(x_list[:,0], x_list[:,1], label="a="+str(a))
    plt.legend()
    
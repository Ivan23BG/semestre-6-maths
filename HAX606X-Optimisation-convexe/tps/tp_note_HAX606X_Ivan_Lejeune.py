# -----
# TP note - descente de gradients et variantes

# -----
#%%
# ----- Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# -----
#%% 
# ----- Paramètres par défaut
# On utilisera sauf mention contraire les paramètres suivants
gamma = 0.01
maxiter = 1000
epsilon = 1e-5

# -----
#%%
# ----- Question 1 - Coder la descente de gradient

# ----- Adaptation de l'algorithme en python avec en sortie la suite des (x_k)
def algo1(grad, x_init, gamma, maxiter, epsilon):
    x = x_init
    x_k = [x]
    for i in range(1, maxiter + 1):
        g = grad(x)
        if np.linalg.norm(g) < epsilon ** 2:
            break
        else:
            x = x - gamma * g
            x_k.append(x)
    return x_k

# -----
#%%
# ----- Verification de l'algorithme sur une fonction simple

# ----- Fonction de test
def f_test(x, y):
    return (x - 1) ** 2 + 3 * (y + 1) ** 2

# ----- Gradient de la fonction de test
def grad_f_test(x):
    return np.array([2 * (x[0] - 1), 6 * (x[1] + 1)])

# ----- Application de l'algorithme
x_init = np.array([0, 0]) # Point initial différent de (1, -1)
x_k = algo1(grad_f_test, x_init, gamma, maxiter, epsilon)
print(f"Résultat de l'algorithme : {x_k[-1]}")

# -----
#%%
# ----- Visualisation de la fonction
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_test(X, Y)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title("Fonction de test")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$f(x, y)$")
plt.show()

# -----
# ----- Fin Question 1
#%%
# ----- Question 2 - Application au cas quadratique

# ----- Fonction quadratique
def f_quad(x):
    if A > 0 and B > 0:
        return (x[1] ** 2) / A + (x[0] ** 2) / B
    else:
        raise ValueError("A et B doivent être strictement positifs")

# ----- Initialisation des paramètres
A = 1
B = 1

# ----- Gradient de la fonction quadratique
def grad_f_quad(x):
    return np.array([2 * x[0] / B, 2 * x[1] / A])

# ----- Application de l'algorithme
x_init = np.array([1, 1]) # Point initial différent de (0, 0)
x_k = algo1(grad_f_quad, x_init, gamma, maxiter, epsilon)
print(f"Résultat de l'algorithme : {x_k[-1]}")

# -----
#%%
# ----- Question 2.1 - Evolution graphique des (x_k) de l'algorithme
# ----- avec a = b = 10, 50, 100
vals = [1, 10, 50, 100]
for val in vals:
    A = val
    B = val
    x_k = algo1(grad_f_quad, x_init, gamma, maxiter, epsilon)
    x_list = np.array(x_k)
    f_quad_list = f_quad(x_list.T)
    # plot with title legend and axis labels
    plt.title(r"Evolution de $f(x_k)$ pour $a = b = 1, 10, 50, 100$")
    plt.xlabel("itération")
    plt.ylabel(r"$f(x_k)$")
    plt.plot(f_quad_list, label=f"a = b = {val}")
    plt.legend()
plt.show()

# ----- Meme chose en echelle logarithmique
for val in vals:
    A = val
    B = val
    x_k = algo1(grad_f_quad, x_init, gamma, maxiter, epsilon)
    x_list = np.array(x_k)
    f_quad_list = f_quad(x_list.T)
    # plot with title legend and axis labels
    plt.yscale("log")
    plt.title(r"Evolution de $f(x_k)$ en echelle log pour $a = b = 1, 10, 50, 100$")
    plt.xlabel("itération")
    plt.ylabel(r"$f(x_k)$")
    plt.plot(f_quad_list, label=f"a = b = {val}")
    plt.legend()
plt.show()
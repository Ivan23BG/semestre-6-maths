# -----
# TP note - descente de gradients et variantes

# -----
#%%
# ----- Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy # pour eviter de modifier les valeurs aux mauvais endroits
import time # pour les comparaisons en temps
from scipy.optimize import minimize

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
def descente_gradient(grad, x_init, gamma, maxiter, epsilon):
    x = x_init
    x_k = [x]
    for _ in range(1, maxiter + 1):
        g = grad(x)
        if np.linalg.norm(g, ord=2) ** 2 <= epsilon ** 2:
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

# ----- Visualisation de la fonction
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_test(X, Y)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title(r"Fonction de test $f_{test}(x, y)$ en 3D")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$f(x, y)$")
plt.show()

# -----
#%%
# ----- Application de l'algorithme
x_init = np.array([0, 0]) # Point initial différent de (1, -1)
x_k = descente_gradient(grad_f_test, x_init, gamma, maxiter, epsilon)
print(f"Résultat de l'algorithme : {x_k[-1]}")

# -----
# ----- Fin Question 1
#%%
# ----- Question 2 - Application au cas quadratique

# ----- Initialisation des paramètres
A = 1
B = 1

# ----- Fonction quadratique
def f_quad(x):
    if A > 0 and B > 0:
        return (x[1] ** 2) / A + (x[0] ** 2) / B
    else:
        raise ValueError("A et B doivent être strictement positifs")

# ----- Gradient de la fonction quadratique
def grad_f_quad(x):
    return np.array([2 * x[0] / B, 2 * x[1] / A])

# ----- Visualisation de la fonction quadratique
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_quad(np.array([X, Y]))
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title(r"Fonction quadratique $f_{\alpha, \beta}(x, y)$ pour $\alpha=\beta=1$ en 3D")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$f(x, y)$")
plt.show()

# -----
#%%
# ----- Application de l'algorithme
x_init = np.array([1, 1]) # Point initial différent de (0, 0)
x_k = descente_gradient(grad_f_quad, x_init, gamma, maxiter, epsilon)
print(f"Résultat de l'algorithme : {x_k[-1]}")

# -----
#%%
# ----- Question 2.1 - Evolution graphique des (x_k) de l'algorithme
# ----- avec a = b = 10, 50, 100
# ----- Evolution de f_quad(x_k) en echelle log
vals = [1, 10, 50, 100]
for val in vals:
    A = val
    B = val
    x_k = descente_gradient(grad_f_quad, x_init, gamma, maxiter, epsilon)
    x_list = np.array(x_k)
    f_quad_list = f_quad(x_list.T)
    # plot with title legend and axis labels
    plt.yscale("log")
    plt.title(r"Evolution de $f_{\alpha, \beta}(x, y)$ en echelle log")
    plt.xlabel("itération")
    plt.ylabel(r"$f_{\alpha, \beta}(x, y)$")
    plt.plot(f_quad_list, label=r"$\alpha=\beta=$" + str(val))
    plt.legend()
plt.show()

# -----
#%%
# ----- Question 2.2 - Lignes de niveaux des fonctions quadratiques et les (x_k)
vals = [1, 10, 50, 100]
X = np.linspace(-2,2,1000)
Y = np.linspace(-2,2,1000)
X, Y = np.meshgrid(X, Y)
x_init = np.array([1, 1])
for val in vals:
    A = val
    B = val
    Z = f_quad(np.array([X, Y]))
    x_k = descente_gradient(grad_f_quad, x_init, gamma, maxiter, epsilon)
    x_list = np.array(x_k)
    Z = f_quad(np.array([X, Y]))
    plt.figure()
    plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r"Lignes de niveau de la fonction $ f_{\alpha, \beta} $")
    plt.colorbar()
    plt.grid(True)
    plt.scatter(x_list[:, 0],x_list[:, 1],color='red',label='Les points itérés',marker='.')
    plt.legend()
plt.show()

# -----
#%%
# ----- Question 2.3 - Remarques
# On remarque pour les vitesses de convergence que plus a est grand
# plus la valeur initiale est proche de celle souhaitée mais plus
# l'algorithme est lent

# ----- Distance a l'optimum en norme l_2 en fonction de a et b
vals = [1, 10, 50, 100]
x_init = np.array([1, 1])
for val in vals:
    A = val
    B = val
    x_k = descente_gradient(grad_f_quad, x_init, gamma, maxiter, epsilon)
    x_list = np.array(x_k)
    norme = [np.linalg.norm(x) for x in x_list]
    plt.plot(norme, label=r"$\alpha=\beta=$" + str(val))
    plt.yscale("log")
    plt.xlabel("itération")
    plt.ylabel("norme")
    plt.title("Distance à l'optimum en norme l_2")
    plt.legend()
plt.show()

# -----
#%%
# ----- Question 2.4 - Cas où a != b
# ----- Affichage de la fonction quadratique
A = 3
B = 15
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_quad(np.array([X, Y]))
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title(r"Fonction quadratique $f_{\alpha, \beta}(x, y)$ pour $\alpha=3, \beta=15$ en 3D")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$f(x, y)$")
plt.show()

# ----- Commentaires
# On remarque que la fonction quadratique est plus aplatie vers x 
# que vers y, ce qui explique la différence de vitesse de convergence

# On retrouve les mêmes problème à minimiser quand la fonction est
# plus aplatie dans une direction que dans l'autre

# -----
# ----- Fin Question 2
#%%
# ----- Question 3 - Descente de gradient par coordonnée

# ----- Question 3.1 - Adaptation de l'algorithme en python avec en sortie la suite des (x_k)
# et même condition d'arrêt
def descente_coord(grad, x_init, gamma, n_iter, epsilon):
    x = x_init
    x_k = [x]
    for i in range(1, n_iter+1):
        for j in range(len(x)):
            g = grad(x)[j]
            if np.linalg.norm(grad(x)[j]) < epsilon ** 2:
                break
            else:
                x[j] = x[j] - gamma * g
                x_k.append(copy.deepcopy(x))
    return x_k

# -----
#%%
# ----- Question 3.2 - Comparaison en temps des deux methodes
x_init = np.array([1, 1])
times = []
for val in vals:
    A = val
    B = val
    t0 = time.perf_counter()
    res = descente_gradient(grad_f_quad, x_init, gamma, maxiter, epsilon)
    t1 = time.perf_counter()-t0
    # print(f"temps gradient: {t1}")

    t0 = time.perf_counter()
    res = descente_coord(grad_f_quad, x_init, gamma, maxiter, epsilon)
    t2 = time.perf_counter()-t0
    # print(f"temps coord: {t2}")
    
    print(f"Rapport : {t2/t1}")

# ----- Commentaires
# On remarque que la méthode de descente de gradient par coordonnée est
# plus rapide que la méthode de descente de gradient classique plus
# a et b sont grands
# -----
#%%
# ----- Question 3.3 - Comparaison des trajectoires des deux méthodes

# ----- Affichage des trajectoires des x_k sur les lignes de niveaux pour le gradient
A = 1
B = 1
x_init = np.array([1, 1])
x_k = descente_gradient(grad_f_quad, x_init, gamma, maxiter, epsilon)
x_list = np.array(x_k)
Z = f_quad(np.array([X, Y]))
plt.figure()
plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r"Lignes de niveau de la fonction $ f_{\alpha, \beta} $")
plt.colorbar()
plt.grid(True)
plt.scatter(x_list[:, 0],x_list[:, 1],color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()

# ----- Affichage des trajectoires des x_k sur les lignes de niveaux pour la coordonnée
A = 1
B = 1
x_init = np.array([1, 1])
x_k = descente_coord(grad_f_quad, x_init, gamma, maxiter, epsilon)
x_list = np.array(x_k)
Z = f_quad(np.array([X, Y]))
plt.figure()
plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r"Lignes de niveau de la fonction $ f_{\alpha, \beta} $")
plt.colorbar()
plt.grid(True)
plt.scatter(x_list[:, 0],x_list[:, 1],color='red',label='Les points itérés',marker='.')
plt.legend()
plt.show()

# -----
# ----- Fin Question 3
#%%
# ----- Question 4 - Avec scipy

# ----- Question 4.1 - methode de descente de gradient
# ----- Commentaires
# Cette méthode n'est pas dans scipy.optimize.minimize

# -----
#%%
# ----- Question 4.a - probleme convexe
# ----- Question 4.a.1 - minimisation de f_quad pour a = b = 20
A = 20
B = 20
x_init = np.array([1, 1])
res1 = minimize(f_quad, x_init, method='nelder-mead',tol=10e-10)
res2 = minimize(f_quad, x_init, method='CG',tol=10e-10)

# ----- Quastion 4.a.2 - comparaison des résultats
print(f"Résultat de la méthode Nelder-Mead : {res1.x}")
print(f"Résultat de la méthode CG : {res2.x}")

# ----- Comparaison du nombre d'itérations
print(f"Nombre d'itérations de la méthode Nelder-Mead : {res1.nfev}")
print(f"Nombre d'itérations de la méthode CG : {res2.nfev}")

# ----- Explications res.success, res.message, res.nfev
# res.success nous dit si la fonction renvoie bien un résultat
# res.message explique pourquoi la fonction a fini
# res.nfev nous donne le nombre d'evaluations effectuées par la fonction

# -----
#%%
# ----- Question 4.b - probleme non convexe

# ----- Question 4.b.1 - representation de Rosenbrock
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

def grad_r(x):
    return np.array([-2 * (1 - x[0]) - 400 * (x[1] - x[0] ** 2) * x[0], 200 * (x[1] - x[0] ** 2)])
# ----- plot de la fonction de Rosenbrock sur [-5, 5]**2
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = rosenbrock(np.array([X, Y]))
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title(r"Fonction de Rosenbrock $f_{Rosenbrock}(x, y)$ en 3D")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$f(x, y)$")
plt.show()

# -----
#%%
# ----- Question 4.b.2 - representation des lignes de niveau
X = np.linspace(0,1.5,1000)
Y = np.linspace(0,1.5,1000)
X, Y = np.meshgrid(X, Y)
Z = rosenbrock(np.array([X, Y]))
plt.figure()
plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r"Lignes de niveau de la fonction de Rosenbrock")
plt.colorbar()
plt.grid(True)
plt.show()

# ----- Commentaires
# Cette fonction est dur a optimiser car elle est très plate
# au "milieu" (en bref, autour de son min) et croit très vite
# en sortant

# -----
#%%
# ----- Question 4.b.3 - minimisation de Rosenbrock

# ----- Paramètres
# On utilisera dans la suite des paramètres adaptés à la fonction de Rosenbrock:
# gamma = 0.001
# maxiter = 500000
# epsilon = 0.01


# ----- Descentes classiques avec pour points: (2,2), (-1,-1), (0.5,1.5)
test1 = descente_gradient(grad_r, (2,2), 0.001, 500000, 0.01)
print("Minimisation avec la descente de gradient classique")
# print(test1)
print(test1[-1])

# -----
test2 = descente_gradient(grad_r, (-1,-1), 0.001, 500000, 0.01)
print("Minimisation avec la descente de gradient classique")
# print(test2)
print(test2[-1])

# -----
test3 = descente_gradient(grad_r, (0.5,1.5), 0.001, 500000, 0.01)
print("Minimisation avec la descente de gradient classique")
# print(test3)
print(test3[-1])

# ----- Descentes par coordonnées avec pour points: (2,2), (-1,-1), (0.5,1.5)
test1_coord = descente_coord(grad_r, [2,2], 0.001, 500000, 0.01)
print("Minimisation avec la descente de gradient par coordonnées")
# print(test1_coord)
print(test1_coord[-1])

# -----
test2_coord = descente_coord(grad_r, [-1,-1], 0.001, 500000, 0.01)
print("Minimisation avec la descente de gradient par coordonnées")
# print(test2_coord)
print(test2_coord[-1])

# -----
test3_coord = descente_coord(grad_r, [0.5,1.5], 0.001, 500000, 0.01)
print("Minimisation avec la descente de gradient par coordonnées")
# print(test3_coord)
print(test3_coord[-1])

# -----
#%%
# ----- Question 4.b.4 - Visualisation des résultats

# ----- Affichage de la descente de gradient sur les lignes de niveau de la fonction de Rosenbrock
X = np.arange(-5,5,0.25)
Y = np.arange(-5,5,0.25)
X, Y = np.meshgrid(X, Y)
Z = rosenbrock(np.array([X, Y]))
plt.figure()
plt.contour(X, Y, Z,20)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Lignes de niveau de la fonction de Rosenbrock')
plt.colorbar(label=r'Valeurs de $f_r(x, y)$')
plt.grid(True)
plt.scatter([v[0] for v in test1],[v[1] for v in test1],color='red',label=r'$x_0=(2, 2)$',marker='.')
plt.scatter([v[0] for v in test2],[v[1] for v in test2],color='blue',label=r'$x_0=(-1, -1)$',marker='.')
plt.scatter([v[0] for v in test3],[v[1] for v in test3],color='green',label=r'$x_0=(-0.5, 1.5)$',marker='.')
plt.legend()
plt.show()

# ----- Affichage de la distance à l’optimum en échelle logarithmique
plt.figure()
plt.yscale("log")
plt.ylabel("Distance à l'optimum")
plt.xlabel("Itérations")
plt.title("Distance à l'optimum en norme l2")
plt.plot([np.linalg.norm((v[0] - 1, v[1]-1),ord=2) for v in test1],label=r'$x_0=(2,2)$')
plt.plot([np.linalg.norm((v[0] - 1, v[1]-1),ord=2) for v in test2],label=r'$x_0=(-1,-1)$')
plt.plot([np.linalg.norm((v[0] - 1, v[1]-1),ord=2) for v in test3],label=r'$x_0=(0.5,1.5)$')
plt.legend()
plt.show()

# -----
# %%
# ----- Question 4.b.5 - Methode de Nelder-Mead

# ----- Minimisation de Rosenbrock avec la méthode de Nelder-Mead
x_init = np.array([2,2])
resu = minimize(rosenbrock, x_init, method='nelder-mead',tol=10e-3)

# ----- Comparaison des résultats
print(f"Iterations nelder-mead: {resu.nfev}")
print(f"Iterations descente de gradient: {len(test1)}")
print(f"Iterations descente coordonnées: {len(test1_coord)}")

# ----- Commentaires
# On remarque que la méthode de Nelder-Mead est plus rapide que
# la descente de gradient et la descente par coordonnées

# -----
# ----- Fin Question 4
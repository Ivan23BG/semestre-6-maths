#%% Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%% Constantes
# on utilisera sauf mention contraire les constantes suivantes
gamma = 0.01
maxiter = 1000
epsilon = 1e-5

#%%
# Question 1 : Coder la descente de gradient
# avec en sortie la suite des x_k
# ensuite tester l'algorithme sur la fonction f_test

#%%
# Algorithme de la descente de gradient
def algo1(grad_f, x_init, gamma, maxiter, epsilon):
    x = x_init
    x_k = [x]
    for i in range(maxiter):
        x = x - gamma * grad_f(x)
        x_k.append(x)
        if np.linalg.norm(grad_f(x)) < epsilon:
            break
    return x_k
#%%
# Fonction de test
def f_test(x_1, x_2):
    return (x_1-1)**2 + 3*(x_2+1)**2

#%%
# gradient de f_test
def grad_f_test(x):
    return np.array([2*(x[0]-1), 6*(x[1]+1)])

#%%
# verification de l'algorithme sur la fonction f_test
x_init = np.array([0, 0])
x_min = algo1(grad_f_test, x_init, gamma, maxiter, epsilon)
print(x_min[-1], "\n",x_min)

#%%
# visualisation de la fonction f_test
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
# tester l'algorithme sur la fonction f_test
x_init = np.array([0, 0])
gamma = 0.01
maxiter = 1000
epsilon = 1e-5
x_min = algo1(grad_f_test, x_init, gamma, maxiter, epsilon)
print(x_min[-1],"\n", x_min)

# %% Question 2 : Application au cas quadratique

# On pose la fonction quadratique f_(a,b)(x,y)=y^2/a+x^2/b
# avec a=b et a=1, puis 10, 50, 100

#%%
# fonction f_ab
def f_ab(x, y):
    global a, b
    if a > 0 and b > 0:
        return y**2/a + x**2/b
    else:
        print("a et b doivent etre positifs")

#%%
# gradient de f_ab
def grad_f_ab(X):
    global a, b
    if a > 0 and b > 0:
        return np.array([2*X[0]/b, 2*X[1]/a])

#%% tester l'algorithme sur la fonction f_ab
a = 1
b = 1
x_init = np.array([1, 1])
x_min = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
print(x_min[-1],"\n", x_min)

#%%
# plot the evolution of f_ab(x_k)
x_list = np.array(x_min)
f_ab_list = f_ab(x_list[:,0], x_list[:,1])
plt.plot(f_ab_list)

#%%
# plot on same graph the evolution of f_ab(x_k) for a=1, 10, 50, 100
x_init = np.array([1, 1])
a_list = [1, 10, 50, 100]
for a in a_list:
    b=a
    x_min = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
    x_list = np.array(x_min)
    f_ab_list = f_ab(x_list[:,0], x_list[:,1])
    # plot
    plt.yscale("log")
    plt.plot(f_ab_list, label="a="+str(a))
    # title
    plt.title("Evolution des f en echelle log pour a=1, 10, 50, 100")
    # plot with legend
    plt.legend()
    print(f"pour a = {a}:\t",x_min[-1],"\n", x_min)

#%% visualisation de la fonction f_ab pour a=1, b=1
a = 1
b = 1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_ab(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#%% visualisation de la fonction f_ab pour a=100, b=100
a = 100
b = 100
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_ab(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
#%% plot countour lines of f_ab and points of x_k
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
for a in a_list:
    b=a
    Z = f_ab(X, Y)
    plt.contour(X, Y, Z, 20)
    plt.colorbar()
    # title
    plt.title("Evolution of contour lines for f_ab")
    # plot the evolution of f_ab(x_k)
    plt.show()
    

#%% on remarque pour les vitesses
# de convergence que plus a est grand
# plus la valeur initiale est proche de celle souhaitée
# mais plus l'algorithme est lent

#%%
# Comparer sur un graphique la distance à l’optimum en norme l_2
# pour chacun des cas avec une échelle logarithmique.



#%% same problem with a != b
# plot on same graph the evolution of f_ab(x_k) for a=1, 10, 50, 100
x_init = np.array([1, 1])
a=3
b=0.5
x_min = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
x_list = np.array(x_min)
f_ab_list = f_ab(x_list[:,0], x_list[:,1])
# plot
plt.yscale("log")
plt.plot(f_ab_list, label="a="+str(a))
# title
plt.title("Evolution des f en echelle log pour a=3, b=0.5")
# plot with legend
plt.legend()
print(f"pour a = {a}:\t",x_min[-1],"\n", x_min)


#%%
# visualisation de la fonction f_ab pour a=3, b=0.5
a = 3
b = 0.5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_ab(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

#%%
# plot countour lines of f_ab and points of x_k
a = 3
b = 0.5
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = f_ab(X, Y)
plt.contour(X, Y, Z, 20)
plt.colorbar()
# title
plt.title("Evolution of f_ab(x_k) for a=3, b=0.5")
# plot the evolution of f_ab(x_k)
x_init = np.array([1, 1])
a=3
b=0.5
x_min = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
x_list = np.array(x_min)
plt.plot(x_list[:,0], x_list[:,1], label="a="+str(a))
plt.legend()


#%%
# Descente par coordonnée

#%%
# Question 3: Descente de gradient par coordonnée

#%%
# Question 3.1: Coder la methode de descente de 
# gradient par coordonnées avec le même critère
# d'arrêt que la descente de gradient classique

def algo2(grad, x_init, gamma, n_iter, epsilon):
    x = x_init
    x_k = [x]
    for i in range(1, n_iter+1):
        for j in range(len(x)):
            x[j] = x[j] - gamma * grad(x)[j]
            if np.linalg.norm(grad(x)[j]) < epsilon:
                break
        x_k.append(x)
        
    return x_k


#%%
# Question 3.2: Comparer la descente de gradient
# classique à la descente de gradient par coordonnées
# pour la fonction f_ab avec differentes valeurs de a=b

# plot la descente de gradient classique
a = 1
b = 1
x_init = np.array([1, 1])
x_min_1 = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
x_list_1 = np.array(x_min_1)
f_ab_list_1 = f_ab(x_list_1[:,0], x_list_1[:,1])
x_min_2 = algo2(grad_f_ab, x_init, gamma, maxiter, epsilon)
x_list_2 = np.array(x_min_2)
f_ab_list_2 = f_ab(x_list_2[:,0], x_list_2[:,1])
# plot
plt.yscale("log")
plt.plot(f_ab_list_1, label="classique")
plt.plot(f_ab_list_2, label="par coordonnee")

# title
plt.title("Evolution des x_k en echelle log pour a=1, b=1")
# plot with legend
plt.legend()
print(f"pour a = {a}:\t",x_min_1[-1],"\n", x_min_1)


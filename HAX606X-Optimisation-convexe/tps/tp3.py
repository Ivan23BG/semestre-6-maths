#%% Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import copy
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
    for i in range(1, maxiter + 1):
        g = grad_f(x)
        if np.linalg.norm(g) < epsilon ** 2:
            break
        else:
            x = x - gamma * g
            x_k.append(x)
    return x_k
#%%
# Fonction de test
def f_test(x_1, x_2):
    return np.square(x_1 - 1) + 3 * np.square(x_2 + 1)

#%%
# gradient de f_test
def grad_f_test(x):
    return np.array([2*(x[0]-1), 6*(x[1]+1)])


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
# verification de l'algorithme sur la fonction f_test
x_init = np.array([0, 0])
x_min = algo1(grad_f_test, x_init, gamma, maxiter, epsilon)
print("Résultat de l'algorithme : ", x_min[-1],"\n", x_min)

# %% Question 2 : Application au cas quadratique

# On pose la fonction quadratique f_(a,b)(x,y)=y^2/a+x^2/b
# avec a=b et a=1, puis 10, 50, 100

#%%
# fonction f_ab
a=1
b=1
def f_ab(x, y):
    global a, b
    if a > 0 and b > 0:
        return np.square(y)/a + np.square(x)/b
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
    b= a
    Z = f_ab(X, Y)
    x_k = np.array(algo1(grad_f_ab, x_init, gamma, maxiter, epsilon))
    plt.scatter(x_k[:,0], x_k[:,1], label="a="+str(a))
    plt.contour(X, Y, Z, 20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    # title
    plt.title("Contour de la fonction")
    # plot the evolution of f_ab(x_k)
    plt.grid()
    plt.legend()
    plt.show()
    

#%% on remarque pour les vitesses
# de convergence que plus a est grand
# plus la valeur initiale est proche de celle souhaitée
# mais plus l'algorithme est lent

#%%
# Comparer sur un graphique la distance à l’optimum en norme l_2
# pour chacun des cas avec une échelle logarithmique.
for a in a_list:
    b= a
    Z = f_ab(X, Y)
    x_k = np.array(algo1(grad_f_ab, x_init, gamma, maxiter, epsilon))
    norme = [np.linalg.norm(x) for x in x_k]
    plt.plot(norme, label="a="+str(a))
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("norme")
    plt.title("Distance à l'optimum en norme l_2")
    plt.legend()
plt.show()


#%% same problem with a != b
# plot on same graph the evolution of f_ab(x_k) for a=1, 10, 50, 100
x_init = np.array([1, 1])
a=3
b=15
Z = f_ab(X, Y)
x_min = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
x_list = np.array(x_min)
plt.scatter(x_list[:,0], x_list[:,1], label="a="+str(a))
plt.contour(X, Y, Z, 20)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
# title
plt.title("Contour de la fonction")
# plot the evolution of f_ab(x_k)
plt.grid()
plt.legend()
plt.show()

#%%
# visualisation de la fonction f_ab pour a=3, b=15
a = 3
b = 15
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
            g = grad(x)[j]
            if np.linalg.norm(grad(x)[j]) < epsilon ** 2:
                break
            else:
                x[j] = x[j] - gamma * g
                x_k.append(copy.deepcopy(x))
    return x_k


#%%
# Question 3.2: Comparer la descente de gradient
# classique à la descente de gradient par coordonnées
# pour la fonction f_ab avec differentes valeurs de a=b

t0 = time.time()
x_init = np.array([1, 1])
a = 1
b = 1
x_k = algo1(grad_f_ab, x_init, gamma, maxiter, epsilon)
t1 = time.time()-t0
print("temps de descente gradient classique",t1)

t0 = time.time()
x_init = np.array([1, 1])
a = 1
b = 1
x_k = algo2(grad_f_ab, x_init, gamma, maxiter, epsilon)
t2 = time.time()-t0
print("temps de descente gradient par coordonées",t2)



print("rapport entre le temps que met gradient 2 et le gradient 1 pour la meme fonction",t2/t1)

plt.title("Évolution graphique de la valeur de l’objectif au cours des itérations de l’algorithme de descente de gradient")


plt.yscale("log")

plt.ylabel("y")
plt.xlabel("x")

plt.plot(np.linspace(1, len(L1), len(L1)), Y1,label = 'alpha =1 classique')
plt.plot(np.linspace(1, len(L3), len(L3)), Y3,label = 'alpha =1 par coordonées')
plt.legend()
plt.show()


plt.yscale("log")
plt.title("Évolution graphique de la valeur de l’objectif au cours des itérations de l’algorithme de descente de gradient")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(np.linspace(1, len(L2), len(L2)), Y2,label = 'alpha =50 classique')
plt.plot(np.linspace(1, len(L4), len(L4)), Y4,label = 'alpha =50 par coordonées')
#print(L3)
plt.legend()


'''
on remarque que la nouvelle descente de gradient est plus lente que 
l'ancienne pour les 2 cas meme 4 fois plus lent en temps
'''



#%%
Z5 = f2(1,1,X,Y)

li5 = algo1(grad1, x_init, gamma, n_iter,0.0001)
x_li, y_li = zip(*li5)  # Séparation des coordonnées x et y
plt.scatter(x_li, y_li, color='red', label='Points de la liste "li"')

plt.contour(X, Y, Z5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de la fonction f2 avec alpha = 1, beta = 1')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.legend()
plt.show()

li6 = algo2( grad1_f, x_init, gamma, n_iter,0.0001)
x_li1, y_li1 = zip(*li6)  # Séparation des coordonnées x et y
plt.scatter(x_li1, y_li1, color='blue', label='Points de la liste "li"')


plt.contour(X, Y, Z5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour de la fonction f2 avec alpha = 1, beta = 1')
plt.colorbar(label='Valeurs de f(x, y)')
plt.grid(True)
plt.legend()
plt.show()

'''
on remarque que il y a 2 traces de points (en bleu) avec la nouvelle 
descente de gradient qui represente chaqune des coordonées
et les 2 traces convergent bien vers 
'''


#%%
def g(x):
    return f2(20, 20, x[0], x[1])
res1 = minimize(g, (1, 1), method="Nelder-Mead")
res2 = minimize(g, (1, 1), method="CG")
print(res1)
print(res2)
# %%
def fr(x, y):
    return (1-x)**2+100*(y-x**2)**2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = fr(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.contour(X, Y, Z)
plt.show()



# %%

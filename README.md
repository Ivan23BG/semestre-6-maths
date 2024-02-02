# Semestre-6-maths


Ce dépôt contient les sources TeX d'un polycopié pour plusieurs cours de L3 maths

# Auteurs

Contributeurs principaux

- Ivan Lejeune

# Source

Ce polycopié de cours repose sur les excellents travaux de Joao Pedro dos Santos, M. de Renzi, M. Marc et M. Charlier

# Compilation

La commande principale de compilation à utiliser est:
```latex
pdflatex %.tex 
```
où `%.tex` est à remplacer par le nom du fichier `.tex` à compiler

Les options utilisables sont:

```latex
-synctex=1
-output-directory=output
```
La première est utile pour une synchronisation avec l'éditeur TexStudio

La seconde est utile pour une clarté de répertoire, quand le temps de compilation n'est pas un problème

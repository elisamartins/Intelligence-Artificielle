include "globals.mzn"; 

int: n;
array[1..n, 1..n] of var 0..1: grid; % var is used in order to take into account the "not-bounded case" _
 
% decision variables: mine or not on the specific cell
array[1..n, 1..n] of var 0..1: x;

% initializing the grid: '_' values on the grid correspond to variables not already set (i.e. the choices you have to make)
constraint forall(i in 1..n, j in 1..n)(x[i,j] = grid[i,j]);

% 1. Pour chaque ligne et colonne, le nombre de cases avec et sans mine doit être identique.
constraint forall(i in 1..n, j in 1..n)(sum(a in 1..n)(x[i,a]) = sum(b in 1..n)(x[b,j]));

% 2. Pour chaque ligne et colonne, Il ne peut pas y avoir plus de deux cases consécutives sans mine.
% Chaque case verifie ses deux voisins du bas sauf les deux dernieres lignes
constraint forall(i in 1..n-2, j in 1..n)(x[i,j] + x[i+1,j] + x[i+2,j] > 0);
% Chaque case verifie ses deux voisins a droite sauf les deux dernieres colonnes
constraint forall(i in 1..n, j in 1..n-2)(x[i,j] + x[i,j+1] + x[i,j+2] > 0);

% 3. Pour chaque ligne et colonne, Il ne peut pas y avoir plus de deux cases consécutives avec mine.
% Chaque case verifie ses deux voisins du bas sauf les deux dernieres lignes
constraint forall(i in 1..n-2, j in 1..n)(x[i,j] + x[i+1,j] + x[i+2,j] < 3);
% Chaque case verifie ses deux voisins a droite sauf les deux dernieres colonnes
constraint forall(i in 1..n, j in 1..n-2)(x[i,j] + x[i,j+1] + x[i,j+2] < 3);

% 4. Il ne peut pas y avoir deux lignes identiques (i.e., ayant des mines exactement au même endroit).
constraint forall(i in 1..n, j in i+1..n)(row(x,i) != row(x,j));

%5. Il ne peut pas y avoir deux colonnes identiques (i.e., ayant des mines exactement au même endroit).

constraint forall(i in 1..n,j in i+1..n)(col(x,i) != col(x,j));

output [
   if j = 1 then "\n" else " " endif ++
     show(x[i,j])
   | i,j in 1..n
];
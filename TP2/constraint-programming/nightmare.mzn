include "globals.mzn"; 
   
% Parameters of the problems
int: n = 5;
array[1..n] of int: left_vision = [2,1,3,2,3];
array[1..n] of int: right_vision = [3,4,1,3,2];
array[1..n] of int: up_vision = [2,1,3,2,2];
array[1..n] of int: down_vision = [3,4,2,1,2];

% Variable: value for each cell
array[1..n,1..n] of var 1..n: city;

% 1. Pour chaque ligne et colonne du terrain, on ne peut pas avoir deux tours ayant la même taille.

constraint forall(i in 1..n)(alldifferent(row(city,i)));
constraint forall(i in 1..n)(alldifferent(col(city,i)));

% 2. Le nombre de tours qu’un observateur situé sur un des bords du terrain est capable de voir est fixé à une certaine valeur. La subtilité réside dans le fait que l’observateur ne peut pas voir une tour s’il se trouve derrière une autre tour de plus grande taille suivant le champ de vision de l’observateur

% Vue de gauche
% Pour chaque ligne a partir de la gauche, on verifie qu'obtient le bon nombre de tours observables en comparant la valeur avec une somme. Pour chaque case a partir de la deuxieme, on verifie si le maximum des cases devant est plus petit que la hauteur de la tour actuelle, si c'est le cas on incremente. On ne verifie pas la premiere tour car elle est forcement visible (on ajoute 1 directement). 

constraint forall(i in 1..n) (
 left_vision[i]= 1 + (sum(j in 2..n)(max(city[1..j-1,i])<city[j,i]))
 );

% Vue de droite
constraint forall(i in 1..n) (
  right_vision[i] = 1 + (sum(j in reverse(1..n-1))(max(city[j+1..n,i])<city[j,i]))
 );

% Vue du haut
constraint forall(i in 1..n) (
 up_vision[i]= 1 + (sum(j in 2..n)(max(city[i,1..j-1])<city[i,j]))
 );

% Vue du bas
constraint forall(i in 1..n) (
  down_vision[i] = 1 + (sum(j in reverse(1..n-1))(max(city[i,j+1..n])<city[i,j]))
 );
output
["city:"]++
[
  if j = 1 then "\n" else " " endif ++
    show(city[j,i])
  | i in 1..n, j in 1..n
];

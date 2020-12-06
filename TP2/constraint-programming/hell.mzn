include "globals.mzn";

int: n_object;
array[1..n_object,1..2] of int: shape; % (x,y) shape of each object
int: max_length;
int: max_width;

% surface totale
array[1..max_length, 1..max_width] of var 1..n_object: surface_totale;

% Tableau des positions des meubles sur la surface (indique la position du point en haut a gauche du meuble)
array[1..n_object, 1..2] of var 1..500: shape_positions;

constraint forall(i in 1..n_object) (shape_positions[i,1] < max_length);
constraint forall(i in 1..n_object) (shape_positions[i,1] < max_width);

constraint forall(i in 1..n_object) (shape_positions[i,1]+shape[i,1] < max_length);
constraint forall(i in 1..n_object) (shape_positions[i,2]+shape[i,2] < max_width);

% Contrainte: aucun meuble un par dessus l'autre 
constraint forall(i in 1..n_object, j in i+1..n_object)(
  (shape_positions[i,1] >= shape_positions[j,1]+ shape[j,1]) \/ 
  (shape_positions[i,2] >= shape_positions[j,2]+ shape[j,2]) \/
  (shape_positions[i,1] + shape[i,1] <= shape_positions[j,1]) \/
  (shape_positions[i,2] + shape[i,2] <= shape_positions[j,2]) 
);

% Cacul de la surface utile
var int: surface_utile = (
  max([shape_positions[i,1]+shape[i,1] | i in 1..n_object]) - min(shape_positions[1..n_object,1])) *
   (max([shape_positions[i,2]+shape[i,2] | i in 1..n_object]) - min(shape_positions[1..n_object,2]));

% On cherche a minimiser la surface utile
solve minimize surface_utile;

output
 ["For shapes: "]++
 [if j = 1 then "\n" else " " endif ++
     show(shape[i,j])
   | i in 1..n_object, j in 1..2]++
 ["\nPositions of top-left corner of shapes are: "]++
 [if j = 1 then "\n" else " " endif ++
     show(shape_positions[i,j])
   | i in 1..n_object, j in 1..2]++
 ["\nWhich gives us a useful surface of: \(surface_utile)\n"];

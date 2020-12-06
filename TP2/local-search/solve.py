
from generator_problem import GeneratorProblem
import random
from random import randrange
import math
class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def solve_naive(self):

        #print("Solve with a naive algorithm")
        #print("All the generators are opened, and the devices are associated to the closest one")

        opened_generators = [1 for _ in range(self.n_generator)]

        assigned_generators = [None for _ in range(self.n_device)]

        for i in range(self.n_device):
            closest_generator = min(range(self.n_generator),
                                    key=lambda j: self.instance.get_distance(self.instance.device_coordinates[i][0],
                                                                      self.instance.device_coordinates[i][1],
                                                                      self.instance.generator_coordinates[j][0],
                                                                      self.instance.generator_coordinates[j][1])
                                    )

            assigned_generators[i] = closest_generator

        self.instance.solution_checker(assigned_generators, opened_generators)
        total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)
        self.instance.plot_solution(assigned_generators, opened_generators, "naive")

        # print("[ASSIGNED-GENERATOR]", assigned_generators)
        # print("[OPENED-GENERATOR]", opened_generators)
        # print("[SOLUTION-COST]", total_cost)

        return assigned_generators, opened_generators

    # Simulated Annealing

    def get_distance(self, i, j):
        return self.instance.get_distance(self.instance.device_coordinates[i][0],
                                    self.instance.device_coordinates[i][1],
                                    self.instance.generator_coordinates[j][0],
                                    self.instance.generator_coordinates[j][1])
    
    def find_closest_generator(self, assigned_generators, opened_generators, device):
        closest_gen = None
        min_distance = None

        for i in range(len(opened_generators)):
            if(opened_generators[i] == 1):
                closest_gen = i
                min_distance = self.get_distance(device, i) 

        for j in range(0, len(opened_generators)):
            if opened_generators[j] == 1:
                distance = self.get_distance(device, j)
                if(distance<min_distance):
                    closest_gen = j
                    min_distance = distance

        return closest_gen

    def open_generator_and_reassign_devices(self, assigned_generators, opened_generators, generator_index):
        opened_generators[generator_index] = 1
        for i in range(len(assigned_generators)):
            assigned_generators[i] = self.find_closest_generator(assigned_generators, opened_generators, i)
        return assigned_generators, opened_generators

    def close_generator_and_reassign_devices(self, assigned_generators, opened_generators, generator_index):
        opened_generators[generator_index] = 0
        for i in range(len(assigned_generators)):
            if(assigned_generators[i] == generator_index):
                assigned_generators[i] = self.find_closest_generator(assigned_generators, opened_generators, i)
        return assigned_generators, opened_generators

    def find_random_neighbor(self, assigned_generators, opened_generators):
        gen_range = list(range(self.n_generator))
        random.shuffle(gen_range)
        index = gen_range.pop()
        if opened_generators[index] == 1 and opened_generators.count(1) > 1:
            return self.close_generator_and_reassign_devices(assigned_generators, opened_generators, index)
        elif opened_generators[index] == 0:
            return self.open_generator_and_reassign_devices(assigned_generators, opened_generators, index)
        else:
            index = gen_range.pop()
            self.open_generator_and_reassign_devices(assigned_generators, opened_generators, index)
    
    def is_accepted(self, delta, temperature):
        return math.exp(-delta / temperature) > random.random()
    
    def solve_simulated_annealing(self, initial_temperature=500, alpha=0.1):
        print("Solve with a simulated annealing algorithm")
        
        # Solution initiale: solution naive
        current_solution = self.solve_naive()
        solution_cost = self.instance.get_solution_cost(current_solution[0], current_solution[1])
        best_solution = current_solution
        best_solution_cost = solution_cost
        
        current_temperature = initial_temperature
        
        while current_temperature > 0:

            # Recherche d'un nouveau voisin
            new_solution = self.find_random_neighbor(current_solution[0], current_solution[1])
            new_cost = self.instance.get_solution_cost(new_solution[0], new_solution[1])   
            
            # Calcul de delta: différence entre les 2 coûts
            delta = new_cost - solution_cost

            # Fonction de sélection: si la nouvelle solution a un coût plus bas, on l'accepte directement. sinon, on l'accepte selon une condition
            if delta <= 0:
                current_solution = new_solution
                solution_cost = new_cost

            elif self.is_accepted(delta, current_temperature): 
                current_solution = new_solution
                solution_cost = new_cost

            if(solution_cost < best_solution_cost):
                best_solution = current_solution
                best_solution_cost = solution_cost
            
            # On réduit la température pour la prochaine itération
            current_temperature -= alpha
        
        # Solution
        print("[ASSIGNED-GENERATOR]", best_solution[0])
        print("[OPENED-GENERATOR]", best_solution[1])
        print("[SOLUTION-COST]", best_solution_cost)
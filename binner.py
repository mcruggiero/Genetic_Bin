from string import ascii_letters

import numpy as np
from numpy import cos, sin, pi
from numpy.random import randint, choice, uniform

import rtree

import pandas as pd
import csv

from matplotlib import pyplot as plt
from IPython.display import Image


class GeneticPacking:
    """
    A genetic packing algorithm to find approximate solutions to the 3D bin
    packing problem
    """

    def __init__(self,
                 num_generations: int = 1000,
                 pop_size: int = 1000,
                 starting_size: int = 1000,
                 size_calc: float = 0,
                 mutation: float = .10,
                 survival: int = 75,
                 keepers: int = 5,
                 csv_location: float = "") -> None:
        """
        Genetic Packing and CSV loader
        ------------------
        :param num_generations:    integer, number of generations to spawn
        :param pop_size:           integer, size of each pop_size
        :param starting_size:      integer, size of starting container
        :param size_calc:          float,   unless 0, takes starting box volumes
                                            and multiples for starting container size
        :param mutation:           float,   % mutation rate for individual
        :param survival:           integer, number of surviving population
        :param keepers:            integer, number of population that survive
                                            after mating
        :param csv_location:       string,  location of packing list .csv
                                            defaults to local folder
        """

        # Basic loading parameters
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.survival = survival
        self.keepers = keepers
        self.mutation = mutation

        # Variables to hold
        self.fitness_values = {}
        self.s_values = []
        self.survivors = {}
        self.next_gen = {}

        # CSV loading options. Defaults to local folder
        if csv_location == "":
            self.purchase = pd.read_csv("purchase_order.csv", index_col=0)
        else:
            self.purchase = pd.read_csv(csv_location, index_col=0)

        # This syntax is not ideal, here we replace the starting_size value
        # by the size_calc factor
        if size_calc != 0:
            self.starting_size = size_calc * (sum(self.purchase["volume"])) ** (1 / 3)
        else:
            self.starting_size = starting_size

        # To maintain the integrity of each chromosome, we will use a dictionary
        # structure. This is the initial population generator
        self.population = {x: self.individual_generator() for x in range(self.pop_size)}

    def gene_generator(self, index) -> dict:
        """
        A random gene generator for an item in CSV
        ------------------
        :param index:              integer, number of gene to grab from.

        """

        entry = dict(self.purchase.loc[index])

        # Similar to Khairuddin, et al. we will be using 5 of our columns
        # for each chromosome, data usage 5 x n, n number of boxes.
        gene = {"index": index,
                "name": entry["name"],
                "weight": entry["weight"],
                "coordinates": uniform(0, self.starting_size, 3),
                "prism_vector": np.array([entry["length"],
                                          entry["width"],
                                          entry["height"]])}

        return self.random_rotation(gene)

    def individual_generator(self) -> dict:
        """
        A quick random individual random generator
        """
        return {x: self.gene_generator(x) for x in self.purchase.index}

    @staticmethod
    def random_rotation(gene) -> dict:
        """
        Applies a random orientation to a gene prism
        ------------------
        :param gene:               dict, gene information with orientation values
        """
        # Roll is x-axis, pitch is y-axis and yaw is z-axis orientation
        # since I am assuming rectangular prisms, we only consider 4 options
        i = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # identity
        r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # roll
        p = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])  # pitch
        y = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])  # yaw

        ###
        # TODO: Consider adding rotations other than Pi/2 ???
        ###

        # Apply random transform to prism vector
        orientations = [i, r, p, y, r * p, r * y, p * y, r * p * y]
        orientation = orientations[choice(len(orientations))]
        gene["prism_vector"] = np.diag(orientation * gene["prism_vector"])

        # This is data needed for our modified overlap calculator
        # rtree coordinates x_min, y_min, z_min, x_max, y_max, z_max
        min_max = [sorted([gene["coordinates"][i],
                           gene["coordinates"][i] + gene["prism_vector"][i]])
                   for i in range(3)]

        gene["rtree_coordinate"] = [min_max[j][i] for i in range(2)
                                    for j in range(len(min_max))]

        return gene

    def fitness_calculator(self, individual) -> float:
        """
        Spacial searching format, Khairuddin, et al. are unclear on how they
        quickly tested if rectangles overlapped. Building a R-tree seems the
        most pythonic option. Returns fitness values as a float.
        ------------------
        :param individual:         integer, the number value for the dict index
        """

        # First initialize the rtree indexer
        p = rtree.index.Property()
        indi = self.population[individual]

        # Next we initialize the properties
        p.dimension = 3
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        idx3d = rtree.index.Index(properties=p)

        # Iterate through all of the genes inside the individual
        i = 0
        while i < len(indi):
            gene_data = self.population[individual][i]

            # If the box intersects another in the set, return infinity fitness
            # since we are minimizing the fitness value, this will effectively kill
            # the individual.
            if len(list(idx3d.intersection(gene_data["rtree_coordinate"]))) > 1:
                return np.Infinity

            idx3d.insert(i, gene_data["rtree_coordinate"])
            i += 1

        # The idx3d bounds returns the format x_min, y_min, ..., y_max, z_max
        bounding_box = idx3d.bounds

        # This list comprehension finds lengths and then multiplies the resulting values
        f_value = np.prod([(bounding_box[i + 3] - bounding_box[i])
                           for i in range(3)])

        # This calculates the surviving individuals from list

        return f_value

    def select(self) -> None:
        """
        Selects top survivors from initial population and populates into dictionary
        """

        # Calculate the fitness of each individual in the population
        self.fitness_values = {}
        for individual in self.population:
            self.fitness_values[individual] = self.fitness_calculator(individual)

        # This can be improved, but for now, we will just sort the dictionary to find survivors
        self.s_values = sorted(self.fitness_values.values())[:self.survival]

        self.survivors = {k: v for k, v in self.fitness_values.items()
                          if v in self.s_values and v != np.Infinity}

        # Sort the survivor dictionary to speed favor best survivors in mating
        self.survivors = dict(sorted(self.survivors.items(),
                                     key=lambda item: item[1]))

    def mutate(self, individual) -> dict:
        """
        Takes population dictionary and randomly reorients boxes with possible
        relocation of object
        """

        ###
        # TODO: Khairuddin, et al were unclear on how often to mutate, might be
        #       fun to experiment with different mutation options
        ###

        # If mutation occurs, first total reset box location
        for gene in individual:
            print(gene)
            if uniform(1) < self.mutation:
                individual[gene]["coordinates"] = uniform(0, self.starting_size, 3)
                individual[gene] = self.random_rotation(individual[gene])
            else:
                individual[gene] = self.random_rotation(individual[gene])

        return individual

    def mate(self) -> None:
        """
        This function takes members of the surviving population and
        mates them by choosing a partition value, makes a new dictionary with
        the values, and then creates the next generation.
        """

        self.next_gen = {}

        # First, pass the keepers into the next generation
        survive_keys = list(self.survivors.keys())[:self.keepers]
        old_num = {k: v for k, v in self.population.items() if k in survive_keys}

        # Renumber keys
        for i, values in enumerate(old_num.items()):
            self.next_gen[i] = values[1]

        # We will be keeping the index values for renumbering
        current_key = self.keepers

        # Next we select one of the top % to mate with the general population
        mating_keys = list(self.survivors.keys())[:int(self.pop_size * self.mutation)]
        gen_keys = set(self.survivors.keys())

        # Choose our lucky couple, partition and mate
        boy_num = choice(mating_keys)
        boy = self.population[boy_num]

        # Numpy can't choose from a set, so making a list from set first
        girl_num = choice(list(set(mating_keys) - {boy_num}))
        girl = self.population[girl_num]

        # This is the splice partition
        splice = int(len(boy))
        self.next_gen[current_key] = child = {k: v for k, v in boy.items() if k < splice}
        child.update({k: v for k, v in girl.items() if k >= splice})
        print(child)
        child = self.mutate(child)  # self.next_gen[current_key]
        current_key += 1

        # Now we splice from the other direction
        self.next_gen[current_key] = child = {k: v for k, v in boy.items() if k >= splice}
        child.update({k: v for k, v in girl.items() if k < splice})
        self.next_gen[current_key] = self.mutate(child)





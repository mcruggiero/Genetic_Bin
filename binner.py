class GeneticPacking:
    """
    A genetic packing algorithm to find approximate solutions to the 3D bin 
    packing problem
    """

    def __init__(self,
                 num_generations: int = 1000,
                 population: int = 1000,
                 starting_size: int = 1000,
                 size_calc: float = 0,
                 survival: int = 75,
                 csv_location: float = "") -> None:
        """
        Genetic Packing and CSV loader
        :param num_generations:    integer, number of generations to spawn
        :param population:         integer, size of each population
        :param starting_size:      integer, size of starting container
        :param size_calc:          float,    unless 0, takes starting box volumes
                                             and multiples for starting container size
        :param survival:           integer, number of surviving population
        :param csv_location:       string,   location of packing list .csv
                                             defaults to local folder
        """

        # Basic loading parameters
        self.num_generations = num_generations
        self.population = population
        self.survival = survival

        # CSV loading options. Defaults to local folder
        if csv_location == "":
            self.purchase = pd.read_csv("purchase_order.csv", index_col=0)
        else:
            self.purchase = pd.read_csv(csv_location, index_col=0)

        if size_calc != 0:
            self.starting_size = size_calc * (sum(self.purchase["volume"])) ** (1 / 3)
        else:
            self.starting_size = starting_size

        # To maintain the integrity of each chromosome, we will use a dictionary
        # structure. This is the initial population generator
        self.population = {x: self.individual_generator()
                           for x in range(self.population)}

        # Calculate the fitness of each individual in the population
        self.fitness_values = {}
        for individual in self.population:
            self.fitness_values[individual] = self.fitness_calculator(individual)

    def gene_generator(self, index) -> dict:
        """
        A random gene generator for an item in CSV
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

    def random_rotation(self, gene) -> dict:
        """
        Applies a random orientation to a gene prism
        :param gene:               dict, gene information with orientation values
        """
        # Roll is x-axis, pitch is y-axis and yaw is z-axis orientation
        # since I am assuming rectangular prisims, we only consider 4 options
        i = identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        r = roll = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        p = pitch = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y = yaw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        ###
        # TODO: Consider adding rotations other than Pi/2 ???
        ###

        # Apply random transform to prism vector
        orientations = [i, r, p, y, r * p, r * y, p * y, r * p * y]
        orientation = orientations[choice(len(orientations))]
        gene["prism_vector"] = np.diag(orientation * gene["prism_vector"])

        # This is data needed for our modified overlap calculator
        # rtree corrdinates x_min, y_min, z_min, x_max, y_max, z_max
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
            # since we are minimizing the fitness value, this will efectively kill
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

        return f_value

    def random_generator(self) -> None:
        """
        """
        pass

    def select(self) -> None:
        """
        """
        pass

    def mate(self) -> None:
        """
        """
        pass

    def mutation(self) -> None:
        pass

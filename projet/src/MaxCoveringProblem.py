from math import ceil
import os
import numpy as np


class MaxCoveringProblem:
    def __init__(self, filename: str):
        self.m, self.n, self.subsets = self.parse_file(filename)
        # The budget k is 1/20 of the number of subsets
        self.k = self.m // 20
        
    def parse_file(self, filename: str):
        with open(filename, "r") as file:
            # Reading n (number of elements in the universe U aka rows) and m (number of subsets aka columns) 
            # Elements are numbered from 1 to n
            n, m = map(int, file.readline().split())

            # Extracting number of elements per line
            elements_per_line = len(list(map(int, file.readline().split())))

            # Skipping the cost of choosing each element (irrelevant to our problem)
            number_of_cost_lines =  ceil(m / elements_per_line) - 1
            for _ in range(number_of_cost_lines):
                file.readline()

            # Reading subsets
            subsets = [set() for _ in range(m)]
            for element in range(n):
                line = file.readline()

                # Skip empty lines
                if not line:
                    continue

                # Reading the number of subsets where element i + 1 appears
                line = list(map(int, line.split()))
                assert len(line) == 1, "Parsing error"
                number_of_occurences = line[0]

                # Filling the occuring subsets with element i
                number_of_lines_per_element = ceil(number_of_occurences / elements_per_line)
                for _ in range(number_of_lines_per_element):
                    line = list(map(int, file.readline().split()))
                    for subset_id in line:
                        # subset_id - 1 to ensure indexing from 0, which is consistent with the indexing of elements (from 0 to n exclusive) 
                        subsets[subset_id - 1].add(element)

            return m, n, subsets
        
    # def evaluate(self, solution: list[int]):
    #     # Calculate the number of unique covered elements for the current position
    #     covered_elements = set().union(*[self.subsets[i] for i in range(self.m) if solution[i] == 1])
    #     return len(covered_elements) 

if __name__ == "__main__":
    dir = "../data/"
    for filename in os.listdir(dir):
        problem  = MaxCoveringProblem(dir + filename)
        print(filename, problem.m, problem.n, problem.subsets, problem.k)

# if __name__ == "__main__":
#     filename = "../testscp.txt"

#     problem  = MaxCoveringProblem(filename)
#     print(f"filename {filename}, m {problem.m}, n {problem.n}, subsets size = m {len(problem.subsets)}, subsets {problem.subsets}, k {problem.k}")
#     print(max([len(subset) for subset in problem.subsets]))
#     print(min([len(subset) for subset in problem.subsets]))
#     print(len(np.unique(np.concatenate([list(subset) for subset in problem.subsets]))))


    
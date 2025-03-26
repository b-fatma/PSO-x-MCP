from math import ceil
import os


class MaxCoveringProblem:
    def __init__(self, filename: str):
        self.m, self.n, self.subsets = self.parse_file(filename)
        # The budget k is 2/3 of the number of subsets
        self.k = 2 * self. m // 3

    def parse_file(self, filename: str):
        with open(filename, "r") as file:
            # Reading m (number of subsets) and n (number of elements in the universe U)
            # Elements are numbered from 1 to n
            m, n = map(int, file.readline().split())

            # Extracting number of elements per line
            elements_per_line = len(list(map(int, file.readline().split())))

            # Reading the cost of choosing each element (irrelevant to our problem)
            number_of_cost_lines =  ceil(n / elements_per_line) - 1
            for _ in range(number_of_cost_lines):
                file.readline()

            # Reading subsets
            subsets = []
            for _ in range(m):
                line = file.readline()

                # Skip empty lines
                if not line:
                    continue

                # Reading the size of the current subset
                line = list(map(int, line.split()))
                assert len(line) == 1, "Parsing error"
                subset_size = line[0]

                # Reading the current subset
                number_of_lines_per_subset = ceil(subset_size / elements_per_line)
                current_subset = [] # could be a set as well ?
                for _ in range(number_of_lines_per_subset):
                    line = list(map(int, file.readline().split()))
                    current_subset.extend(line)
                subsets.append(current_subset)

            return m, n, subsets
        
    # def evaluate(self, solution: list[int]):
    #     # Calculate the number of unique covered elements for the current position
    #     covered_elements = set().union(*[self.subsets[i] for i in range(self.m) if solution[i] == 1])
    #     return len(covered_elements) 

# if __name__ == "__main__":
#     dir = "../data/"
#     for filename in os.listdir(dir):
#         problem  = MaxCoveringProblem(dir + filename)
#         print(filename, problem.m, problem.n, len(problem.subsets), problem.k)

    
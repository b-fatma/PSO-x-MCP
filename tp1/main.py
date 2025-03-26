import random

class KnapsackProblem:
    def __init__(self, n):
        self.n = n
        self.W = random.randint(50, 100)  # Capacity between 50 and 100
        self.items = []

        for _ in range(self.n):
            value = random.randint(5, 25)   # Value between 5 and 25
            weight = random.randint(4, 24)  # Weight between 4 and 24
            self.items.append({'value': value, 'weight': weight})

    # Print the instance details
    def print_instance(self):
        print(f"Knapsack Capacity: {self.W}")
        print("Items (Value, Weight):")
        for i, item in enumerate(self.items):
            print(f"Item {i}: ({item['value']}, {item['weight']})")

    # Generate a random solution
    def generate_solution(self):
        return [random.randint(0, 1) for _ in range(self.n)]

    # Check if a solution is valid
    def is_valid_solution(self, solution):
        total_weight = sum(self.items[i]['weight'] for i in range(self.n) if solution[i] == 1)
        return total_weight <= self.W

    # Evaluate the total value of the solution
    def evaluate_solution(self, solution):
        total_value = sum(self.items[i]['value'] for i in range(self.n) if solution[i] == 1)
        return total_value

    # Print the solution
    def print_solution(self, solution):
        print(f"Solution: {solution}")
        print("Items in Solution (Value, Weight):")
        for i in range(self.n):
            if solution[i] == 1:
                print(f"Item {i}: ({self.items[i]['value']}, {self.items[i]['weight']})")


if __name__ == "__main__":
    random.seed()  

    instance = KnapsackProblem(10)
    instance.print_instance()

    print("Generate random solutions, check their validity, and evaluate them.")
    for _ in range(3):
        solution = instance.generate_solution()
        instance.print_solution(solution)
        print("The solution is: ", end="")
        if instance.is_valid_solution(solution):
            print("valid")
            print(f"Evaluation: {instance.evaluate_solution(solution)}")
        else:
            print("not valid")

import random
import time
import matplotlib.pyplot as plt

class KnapsackProblem:
    def __init__(self, n):
        self.n = n
        self.W = random.randint(50, 100)  
        self.items = [{'value': random.randint(5, 25), 'weight': random.randint(10, 30)} for _ in range(n)]
    
    def print_instance(self):
        print(f"Knapsack Capacity: {self.W}")
        print("Items (Value, Weight):")
        for i, item in enumerate(self.items):
            print(f"Item {i}: ({item['value']}, {item['weight']})")

    def solve_dfs(self, index=0, current_weight=0, current_value=0, selected=None, depth=0, verbose=False):
        if selected is None:
            selected = [0] * self.n  

        if verbose:
            print("  " * depth + f"Node: index={index}, weight={current_weight}, value={current_value}, selected={selected}")
        
        if index == self.n:
            return current_value, selected

        best_value, best_selection = self.solve_dfs(index + 1, current_weight, current_value, selected[:], depth + 1, verbose)

        if current_weight + self.items[index]['weight'] <= self.W:
            new_selected = selected[:]
            new_selected[index] = 1  
            new_value, new_selection = self.solve_dfs(
                index + 1, 
                current_weight + self.items[index]['weight'], 
                current_value + self.items[index]['value'], 
                new_selected,
                depth + 1,
                verbose
            )
            
            if new_value > best_value:
                best_value, best_selection = new_value, new_selection

        return best_value, best_selection
    
    

def complexity_test():
    sizes = range(10, 1000, 10)
    times = []
    valid_sizes = []
    timeout_limit = 600 # 10 min

    for n in sizes:
        total_time = 0
        runs = 10  
        print(f"Testing n={n}...")

        for _ in range(runs):
            instance = KnapsackProblem(n)
            start_time = time.time()

            _, _ = instance.solve_dfs()
            elapsed_time = time.time() - start_time
            print(f"    current run took {elapsed_time:.5f}s.")

            if elapsed_time > timeout_limit:
                print(f"Skipping n={n} (took {elapsed_time:.2f}s, over {timeout_limit/60} min).")
                break

            total_time += elapsed_time

        else:  
            avg_time = total_time / runs
            times.append(avg_time)
            valid_sizes.append(n)
            print(f"n={n}: Avg Time = {avg_time:.5f} seconds")
            continue
        
        break  

    if valid_sizes:
        plt.plot(valid_sizes, times, 'go-', label='Avg Time Taken')
        plt.xlabel('Number of Items (n)')
        plt.ylabel('Avg Time Taken (seconds)')
        plt.title('Knapsack DFS Performance')
        plt.grid(True)
        plt.legend()
        plt.savefig("knapsack_dfs_time.png")
        plt.show()
    else:
        print("No valid data points to plot.")



if __name__ == "__main__":
    random.seed()
    
    instance = KnapsackProblem(3)
    instance.print_instance()

   
    print("\nExploring DFS Tree:")
    max_value, solution = instance.solve_dfs(verbose=True)
    
    print(f"\nMaximum Knapsack Value: {max_value}")
    print(f"Best Solution: {solution}")

    print("\nItems in the Optimal Solution:")
    for i in range(instance.n):
        if solution[i] == 1:
            print(f"Item {i}: (Value={instance.items[i]['value']}, Weight={instance.items[i]['weight']})")

    # complexity_test()
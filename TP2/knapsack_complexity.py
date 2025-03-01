import random
import time
import matplotlib.pyplot as plt

class KnapsackProblem:
    def __init__(self, n):
        self.n = n
        self.W = random.randint(50, 100)
        self.items = []

        for _ in range(self.n):
            value = random.randint(5, 25)
            weight = random.randint(10, 30)
            self.items.append({'value': value, 'weight': weight})

    def knapsack_dfs(self, index=0, current_weight=0, current_value=0, selected=None):
        if selected is None:
            selected = [0] * self.n

        if index == self.n:
            return current_value, selected

        best_value, best_selection = self.knapsack_dfs(index + 1, current_weight, current_value, selected[:])

        if current_weight + self.items[index]['weight'] <= self.W:
            new_selected = selected[:]
            new_selected[index] = 1
            new_value, new_selection = self.knapsack_dfs(
                index + 1,
                current_weight + self.items[index]['weight'],
                current_value + self.items[index]['value'],
                new_selected
            )
            
            if new_value > best_value:
                best_value, best_selection = new_value, new_selection

        return best_value, best_selection

if __name__ == "__main__":
    random.seed()
    sizes = [5,10, 15, 20, 25, 30, 35, 40, 45, 50]
    times = []
    
    for n in sizes:
        total_time = 0
        runs = 10  # Number of runs to calculate average
        for _ in range(runs):
            instance = KnapsackProblem(n)
            start_time = time.time()
            max_value, solution = instance.knapsack_dfs()
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            total_time += elapsed_time
        
        avg_time = total_time / runs
        times.append(avg_time)
        print(f"n={n}: Avg Time taken = {avg_time:.5f} seconds")
 
plt.plot(sizes, times, 'go-', label='Avg Time Taken')
for i, txt in enumerate(times):
    plt.text(sizes[i], times[i], f"{txt:.5f}", fontsize=10, verticalalignment='bottom')
    
plt.xlabel('Number of Items (n)')
plt.ylabel('Avg Time Taken (seconds)')
plt.title('Knapsack DFS Performance')
plt.grid(True)
plt.legend()
plt.savefig("knapsack_dfs_time.png")
plt.show()
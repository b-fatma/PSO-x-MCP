import random

class KnapsackProblem:
    def __init__(self):
        self.n = random.randint(3, 5)  
        self.W = random.randint(50, 100)  
        self.items = []

        for _ in range(self.n):
            value = random.randint(5, 25)   
            weight = random.randint(10, 30)  
            self.items.append({'value': value, 'weight': weight})
    
    def print_instance(self):
        print(f"Knapsack Capacity: {self.W}")
        print("Items (Value, Weight):")
        for i, item in enumerate(self.items):
            print(f"Item {i}: ({item['value']}, {item['weight']})")

    def knapsack_dfs(self, index=0, current_weight=0, current_value=0, selected=None, depth=0):
        if selected is None:
            selected = [0] * self.n  

        #print current state
        print("  " * depth + f"Node: index={index}, weight={current_weight}, value={current_value}, selected={selected}")
        
        if index == self.n:
            return current_value, selected

        best_value, best_selection = self.knapsack_dfs(index + 1, current_weight, current_value, selected[:], depth + 1)

        if current_weight + self.items[index]['weight'] <= self.W:
            new_selected = selected[:]
            new_selected[index] = 1  
            new_value, new_selection = self.knapsack_dfs(
                index + 1, 
                current_weight + self.items[index]['weight'], 
                current_value + self.items[index]['value'], 
                new_selected,
                depth + 1
            )
            
            if new_value > best_value:
                best_value, best_selection = new_value, new_selection

        return best_value, best_selection

if __name__ == "__main__":
    random.seed()
    
    instance = KnapsackProblem()
    instance.print_instance()

   
    print("\nExploring DFS Tree:")
    max_value, solution = instance.knapsack_dfs()
    
    print(f"\nMaximum Knapsack Value: {max_value}")
    print(f"Best Solution: {solution}")

    print("\nItems in the Optimal Solution:")
    for i in range(instance.n):
        if solution[i] == 1:
            print(f"Item {i}: (Value={instance.items[i]['value']}, Weight={instance.items[i]['weight']})")
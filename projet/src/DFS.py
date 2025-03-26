from MaxCoveringProblem import MaxCoveringProblem
import time


class DFS:
    def __init__(self, problem, time_limit=10):  
        self.problem = problem
        self.time_limit = time_limit
        self.best_valid_fitness = float('-inf')  # Stores the best valid solution found
        self.best_valid_selection = None
        self.best_fitness = float('-inf') # Stores the best partial (valid/invalid) solution
        self.best_selection = [0] * self.problem.m 

    
    def solve(self, index=0, current_budget=0, covered_elements=set(), selected=None, verbose=False):
        if selected is None:
            selected = [0] * self.problem.m 

        if verbose:
            print(f"Depth: {index}, Budget Used: {current_budget}/{self.problem.k}, Covered Elements: {len(covered_elements)}, Selected: {selected}")
        
        # Base case: k subsets selected
        if current_budget == self.problem.k:
            return len(covered_elements), selected
        
        # If we've considered all subsets and selected less than k, return worst case
        if index == self.problem.m:
            return float('-inf'), selected  # -inf means invalid solution
        
        # Case 1: skip the current subset
        best_fitness, best_selection = self.solve(index + 1, current_budget, covered_elements, selected[:], verbose)

        # Case 2: select the current subset (only if we haven't reached k)
        if current_budget < self.problem.k:
            new_selected = selected[:]
            new_selected[index] = 1  
            new_fitness, new_selection = self.solve(
                index + 1, 
                current_budget + 1, 
                covered_elements.union(self.problem.subsets[index]),  
                new_selected,
                verbose
            )
            
            # Keep the better selection
            if new_fitness > best_fitness:
                best_fitness, best_selection = new_fitness, new_selection

        return best_fitness, best_selection
    

    def solve_time_bound(self, index=0, current_budget=0, covered_elements=set(), selected=None, start_time=None, verbose=False):
        """
        A depth-first search that stops execution if the time limit is exceeded.
        Returns:
            best_fitness (int): Maximum number of elements covered.
            best_selection (list): Subset selection achieving best_fitness.
            completed (bool): True if DFS finished, False if time limit was reached.
        """
        if start_time is None:
            start_time = time.time()

        if selected is None:
            selected = [0] * self.problem.m  

        # Time limit check
        # We prioritize validity of solutions (exactly k subsets) over fitness of solutions
        if time.time() - start_time >= self.time_limit:
            if self.best_valid_selection is not None:
                return self.best_valid_fitness, self.best_valid_selection, False  
            return self.best_fitness, self.best_selection, False  

        if verbose:
            print(f"Depth: {index}, Budget Used: {current_budget}/{self.problem.k}, "
                f"Covered Elements: {len(covered_elements)}, Selected: {selected}")

        # Base case: k subsets selected
        if current_budget == self.problem.k:
            if len(covered_elements) > self.best_valid_fitness:
                self.best_valid_fitness = len(covered_elements)
                self.best_valid_selection = selected[:]  
            
            if len(covered_elements) > self.best_fitness:
                self.best_fitness = len(covered_elements)
                self.best_selection = selected[:]

            return len(covered_elements), selected, True  
        
        # If all subsets are considered but k is not reached, return best partial solution
        if index == self.problem.m:
            if len(covered_elements) > self.best_fitness:
                self.best_fitness = len(covered_elements)
                self.best_selection = selected[:]
            return self.best_fitness, self.best_selection, True  

        # Case 1: Skip the current subset
        new_fitness, new_selection, completed_skip = self.solve_time_bound(
            index + 1, current_budget, covered_elements, selected[:], start_time, verbose
        )

        # Update the best solution if skipping this subset gives a better result
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_selection = new_selection[:]

        # Case 2: Select the current subset (only if we haven't reached k)
        completed = completed_skip  
        if current_budget < self.problem.k:
            new_selected = selected[:]
            new_selected[index] = 1  
            new_fitness, new_selection, completed_select = self.solve_time_bound(
                index + 1, 
                current_budget + 1, 
                covered_elements.union(self.problem.subsets[index]),  
                new_selected,
                start_time, 
                verbose
            )
            
            # Keep track of the best solution found (even if timeout occurs)
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_selection = new_selection[:]

            # Update the completion status
            completed = completed_skip and completed_select  

        return self.best_fitness, self.best_selection, completed  
    

    def solve_time_bound_(self, index=0, current_budget=0, covered_elements=set(), selected=None, start_time=None, verbose=False):
        """
        A depth-first search that stops execution if the time limit is exceeded.
        Returns:
            best_fitness (int): Maximum number of elements covered.
            best_selection (list): Subset selection achieving best_fitness.
            completed (bool): True if DFS finished, False if time limit was reached.
            execution_time (float): Total time taken by DFS.
        """
        if start_time is None:
            start_time = time.time()

        if selected is None:
            selected = [0] * self.problem.m  

        elapsed_time = time.time() - start_time  

        # Time limit check
        if elapsed_time >= self.time_limit:
            if self.best_valid_selection is not None:
                return self.best_valid_fitness, self.best_valid_selection, False, elapsed_time  
            return self.best_fitness, self.best_selection, False, elapsed_time  

        if verbose:
            print(f"Depth: {index}, Budget Used: {current_budget}/{self.problem.k}, "
                f"Covered Elements: {len(covered_elements)}, Selected: {selected}")

        # Base case: k subsets selected
        if current_budget == self.problem.k:
            if len(covered_elements) > self.best_valid_fitness:
                self.best_valid_fitness = len(covered_elements)
                self.best_valid_selection = selected[:]  
            
            if len(covered_elements) > self.best_fitness:
                self.best_fitness = len(covered_elements)
                self.best_selection = selected[:]

            return len(covered_elements), selected, True, time.time() - start_time  

        # If all subsets are considered but k is not reached, return best partial solution
        if index == self.problem.m:
            if len(covered_elements) > self.best_fitness:
                self.best_fitness = len(covered_elements)
                self.best_selection = selected[:]
            return self.best_fitness, self.best_selection, True, time.time() - start_time  

        # Case 1: Skip the current subset
        new_fitness, new_selection, completed_skip, _ = self.solve_time_bound_(
            index + 1, current_budget, covered_elements, selected[:], start_time, verbose
        )

        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_selection = new_selection[:]

        # Case 2: Select the current subset (only if we haven't reached k)
        completed = completed_skip  
        if current_budget < self.problem.k:
            new_selected = selected[:]
            new_selected[index] = 1  
            new_fitness, new_selection, completed_select, _ = self.solve_time_bound_(
                index + 1, 
                current_budget + 1, 
                covered_elements.union(self.problem.subsets[index]),  
                new_selected,
                start_time, 
                verbose
            )
            
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_selection = new_selection[:]

            completed = completed_skip and completed_select  

        return self.best_fitness, self.best_selection, completed, time.time() - start_time  

# if __name__ == "__main__":
#     problem = MaxCoveringProblem("../data/scp41.txt")
#     dfs_solver  = DFS(problem)
#     best_fitness, best_selection, completed = dfs_solver.solve_time_bound(verbose=True)
#     print(f"Fitness = {best_fitness}, Used budget = {sum(best_selection)} / {problem.k}, Completed? {completed}")
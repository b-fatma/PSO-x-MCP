#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

// Struct to represent an item
struct Item {
    int value;
    int weight;
};

class KnapsackProblem {
    int n;
    vector<Item> items;
    int W;

    public:

    KnapsackProblem(){
        n = 1 + rand() % 10;
        W = 50 + rand() % 51; // Capacity between 50 and 100
        for (int i = 0; i < n; ++i) {
            int value = 5 + rand() % 21; // Value between 5 and 25
            int weight = 4 + rand() % 21; // Weight between 4 and 24
            items.push_back({value, weight});
        }
    }

    void print_instance(){
        cout << "Knapsack Capacity: " << W << "\n";
        cout << "Items (Value, Weight):\n";
        for (int i = 0; i < items.size(); ++i) {
            cout << "Item " << i << ": (" << items[i].value << ", " << items[i].weight << ")\n";
        }
    }

    vector<int> generate_solution(){
        vector<int> sol(n);
        for(int i = 0; i < n; i++){
            sol[i] = rand() % 2;
        }
        return sol;
    }

    bool is_valid_solution(const vector<int>& solution) {
        int total_weight = 0;
        for (int i = 0; i < n; i++) {
            if (solution[i] == 1) {
                total_weight += items[i].weight;
            }
        }
        return total_weight <= W;
    }

    int evaluate_solution(const vector<int>& solution) {
        int total_value = 0;
        for (int i = 0; i < n; i++) {
            if (solution[i] == 1) {
                total_value += items[i].value;
            }
        }
        return total_value;
    }

    void print_solution(const vector<int>& solution) {
        cout << "Solution: ";
        for (int i = 0; i < n; i++) {
            cout << solution[i];
            if(i != n - 1) cout << ", ";
            else cout << endl;
        }
        cout << "Solution (Value, Weight):\n";
        for (int i = 0; i < n; i++) {
            if (solution[i] == 1) {
                cout << "Item " << i << ": (" << items[i].value << ", " << items[i].weight << ")\n";
            }
        }
    }
};

int main() {

    srand(time(0));

    KnapsackProblem* instance  = new KnapsackProblem();

    instance->print_instance();

    cout << "Generate random solutions, check their validity, and evaluate them." << endl;
    for(int i = 0; i < 3; i++){
        vector<int> solution = instance->generate_solution();
        instance->print_solution(solution);
        cout << "The solution is: ";
        if(instance->is_valid_solution(solution)) {
            cout << "valid" << endl;
            cout << "Evaluation: " << instance->evaluate_solution(solution) << endl;
        }
        else{
            cout << "not valid" << endl;
        }
    }
}

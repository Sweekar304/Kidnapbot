#!/usr/bin/env python3

from sys import maxsize
from itertools import permutations
import numpy as np

class Travelling_salesman:
    def __init__(self, objects, start, graph) -> None:
        self.objects = objects
        self.s = 0
        self.cost = 0
        self.graph = graph
        self.best_route = []

    def path_len(self, path):
        return sum(self.graph[i][j] for i, j in zip(path, path[1:]))

    def solve(self):
        # print(".......................")
        # print(self.graph)
        # print(".......................")
        to_visit = set(range(len(self.graph)))

        # Current state {(node, visited_nodes): shortest_path}
        state = {(i, frozenset([0, i])): [0, i] for i in range(1, len(self.graph[0]))}

        for _ in range(len(self.graph) - 2):
            next_state = {}
            for position, path in state.items():
                current_node, visited = position

                # Check all nodes that haven't been visited so far
                for node in to_visit - visited:
                    new_path = path + [node]
                    new_pos = (node, frozenset(new_path))

                    # Update if (current node, visited) is not in next state or we found shorter path
                    if new_pos not in next_state or self.path_len(new_path) < self.path_len(next_state[new_pos]):
                        next_state[new_pos] = new_path

            state = next_state
        
        shortest = min((path + [0] for path in state.values()), key=self.path_len)

        return shortest

# if __name__ == '__main__':
#     objects = {'1': (10, 15), '2': (2, 2), '3': (-12, 9)}
#     start = (0, 0)
#     graph = []
#     salesman = Travelling_salesman(objects, start, graph)
#     min_path, best_route = salesman.solve()
#     print(min_path)
#     print(best_route)

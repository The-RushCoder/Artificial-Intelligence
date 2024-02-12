#Method-1
"""
adjacency_list = {
    'S': [('A', 1), ('B', 4)],
    'A': [('B', 2), ('C', 5), ('D', 12)],
    'B': [('C', 2)],
    'C': [('D', 3)],
    'D': []
}


class Graph:

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def h(self, n):
        H = {
            'S': 7,
            'A': 6,
            'B': 2,
            'C': 1,
            'D': 0
        }
        return H[n]

    def a_star_algorithm(self, start_node, stop_node):
        open_list = set([start_node])
        closed_list = set([])

        g = {}
        g[start_node] = 0
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            for (m, weight) in self.get_neighbors(n):

                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight


                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            open_list.remove(n)
            closed_list.add(n)
            print(weight);

        print('Path does not exist!')
        return None


graph1 = Graph(adjacency_list)
graph1.a_star_algorithm('S', 'D')
"""


#Method-2
import math
from queue import PriorityQueue

coords = {}  # node id is the key
adjlist = {}  # node id is the key

with open('input2.txt', 'r') as f:
    V = int(f.readline())
    for i in range(V):
        strs = f.readline().split()
        nid, x, y = strs[0], int(strs[1]), int(strs[2])
        coords[nid] = (x, y)  # x, y kept as a tuple
        adjlist[nid] = []  # create empty list for each node's adjnodes

    E = int(f.readline())
    for i in range(E):
        strs = f.readline().split()
        n1, n2, c = strs[0], strs[1], int(strs[2])
        adjlist[n1].append((n2, c))  # (n2, c) tuple

    startnid = f.readline().rstrip()
    goalnid = f.readline().rstrip()

print('graph')
for nid in adjlist:
    print(nid, coords[nid], '--->', adjlist[nid])
    for tup in adjlist[nid]:
        print('\t', tup[0], tup[1])
print('start', startnid, 'goal', goalnid)


def heuristic(node_id, goal_node):
    # Calculate Euclidean distance between node_id and goal_node
    x1, y1 = coords[node_id]
    x2, y2 = coords[goal_node]
    euclidean_dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return euclidean_dist


class State:
    def __init__(self, node_id, parent=None, g=0, f=0):
        self.node_id = node_id
        self.parent = parent
        self.g = g  # cost from start to current node
        self.f = f  # estimated total cost from start to goal through current node

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.node_id == other.node_id


start_node = State(startnid, None, 0, heuristic(startnid, goalnid))
minQ = PriorityQueue()
minQ.put(start_node)

visited = set()

while not minQ.empty():
    curr_state = minQ.get()
    visited.add(curr_state.node_id)

    if curr_state.node_id == goalnid:
        # Goal reached, construct the solution path
        solution = []
        optimal_cost = curr_state.g  # Store the optimal cost before the loop exits
        while curr_state:
            solution.append(curr_state.node_id)
            curr_state = curr_state.parent
        solution.reverse()
        print("Solution path:", ' – '.join(solution))  # Print the solution path as a string
        print("Solution cost:", optimal_cost)  # Print the stored optimal cost
        break

    for adj_node, cost in adjlist[curr_state.node_id]:
        if adj_node not in visited:
            g = curr_state.g + cost  # Update the cost from start to the adjacent node
            h = heuristic(adj_node, goalnid)
            f = g + h  # Correct calculation of total cost
            new_state = State(adj_node, curr_state, g, f)
            minQ.put(new_state)

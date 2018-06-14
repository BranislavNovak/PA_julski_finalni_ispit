import sys
import math
from enum import Enum


# -------------------------------------------------- classes:
class Vertex:
    """
    Graph vertex: A graph vertex (node) with data
    """
    def __init__(self, p = None, d1 = None, d2 = None):
        """
        Vertex constructor 
        @param color, parent, auxilary data1, auxilary data2
        """
        self.p = p
        self.d1 = d1
        self.d2 = d2
        self.list = list()
        self.color = None
        self.childs = list()

class Edge:
    def __init__(self, source = None, destination = None, weight = None):
        self.source = source;
        self.destination = destination
        self.weight = weight

class VertexColor(Enum):
        BLACK = 0
        GRAY = 127
        WHITE = 255	

class Tree:
    def __init__(self, r = None):
        self.root = r

# -------------------------------------------------- functions:
def make_graph():
    graph = list()
    
    A = Vertex(d1 = 'A')        # 3 edges
    B = Vertex(d1 = 'B')        # 1 edges
    C = Vertex(d1 = 'C')        # 3 edges
    D = Vertex(d1 = 'D')        # 2 edges
    E = Vertex(d1 = 'E')        # 4 edges
    F = Vertex(d1 = 'F')        # 2 edges
    G = Vertex(d1 = 'G')        # 3 edges
    H = Vertex(d1 = 'H')        # 0 edges
    I = Vertex(d1 = 'I')        # 1 edges

    A.list.append(Edge(A, B, 15))
    A.list.append(Edge(A, C, 13))
    A.list.append(Edge(A, D, 5))

    B.list.append(Edge(B, H, 12))

    C.list.append(Edge(C, F, 6))
    C.list.append(Edge(C, B, 2))
    C.list.append(Edge(C, D, 18))
    
    D.list.append(Edge(D, E, 4))
    D.list.append(Edge(D, I, 99))
    
    E.list.append(Edge(E, C, 3))
    E.list.append(Edge(E, F, 1))
    E.list.append(Edge(E, G, 9))
    E.list.append(Edge(E, I, 14))
    
    F.list.append(Edge(F, B, 8))
    F.list.append(Edge(F, H, 17))
    
    G.list.append(Edge(G, F, 16))
    G.list.append(Edge(G, H, 7))
    G.list.append(Edge(G, I, 10))
    
    I.list.append(Edge(I, H, 11))
        
    graph.append(A)
    graph.append(B)
    graph.append(C)
    graph.append(D)
    graph.append(E)
    graph.append(F)
    graph.append(G)
    graph.append(H)
    graph.append(I)

    return graph

# Using Bellman_Ford algorithm for finding shortest path (pseudo code)
def initialize_single_source(G,s):
    for v in G:
        v.d2 = math.inf
        v.p = None
    s.d2 = 0

def relax(u, v, w):    
    if v.d2 > u.d2 + w:
        v.d2 = u.d2 + w
        v.p = u

def BellMan_Ford(G,w,s):    
    initialize_single_source(G,s)
    for i in G:
        for j in i.list:
            relax(j.source, j.destination, j.weight)
    for i in G:
        for j in i.list:
            if j.destination.d2 > j.source.d2 + j.weight:
                return False
    return True 
   
def shortest_path(graph, B):
    shortest_path_list = list()
    # Call of BellMan_Ford function
    BellMan_Ford(graph, 0, graph[0])
    w = B.d2
    parent = B.p
    shortest_path_list.append(B)

    while parent != graph[0]:                       # Comparing with graph[0] cause it is vertex A.
        shortest_path_list.append(parent)
        parent = parent.p
    shortest_path_list.append(graph[0])
    shortest_path_list = shortest_path_list[::-1]

    return shortest_path_list,w
    
# BFS (pseudo code)
def BFS(graph, start_node):
    d = dict()
    
    for u in graph:
        u.color = VertexColor.WHITE
        u.d2 = math.inf
        u.p = None

    start_node.color = VertexColor.GRAY
    start_node.d2 = 0
    start_node.p = None
    Q = list()
    Q.append(start_node)
    while len(Q) is not 0:
        k = Q.pop(0)
        for v in k.list:
            if v.destination.color == VertexColor.WHITE:
                v.destination.color = VertexColor.GRAY
                v.destination.d2 = k.d2 + 1
                v.destination.p = k
                d[v.destination.d1] = v.destination.d2        # Setting keys and their value.
                Q.append(v.destination)
        k.color = VertexColor.BLACK

    d[start_node.d1] = 0                                      # Setting start node to 0 weight
    return d

# BFS_tree
def BFS_tree(graph, start_node):    
    
    for u in graph:
        u.color = VertexColor.WHITE
        u.d2 = math.inf
        u.p = None
    
    start_node.color = VertexColor.GRAY
    start_node.d2 = 0
    start_node.p = None
    Q = list()
    Q.append(start_node)
    while len(Q) is not 0:
        k = Q.pop(0)
        for v in k.list:
            if v.destination.color == VertexColor.WHITE:
                v.destination.color = VertexColor.GRAY
                v.destination.d2 = k.d2 + 1
                v.destination.p = k
                
                k.childs.append(v)
                    
                Q.append(v.destination)
        k.color = VertexColor.BLACK

    t = Tree(start_node)
    return t
 
# ------------------------------------------------------ HELP FUNCTIONS 

# Help function for check if everything is ok.
def print_graph(graph):        
    for k in graph:
        print("***", k.d1)
        for n in k.list:
            print("---> ", n.destination.d1, "with edge", n.weight)

# Help function for check if everything is ok.   
def print_shortest_path(shortest_path_list, weight):
    counter = 0
    for i in shortest_path_list:
        print("--- step", counter, "to vertex", i.d1)
        counter += 1
    print("Full weight of path: ", weight) 

def print_tree(tree):
    root = tree.root
    print("\n")
    for i in root.childs:
        print(i.destination.d1, "parent is:", i.destination.p.d1)
        for j in i.destination.childs:
            print(j.destination.d1, "parent is:", j.destination.p.d1)
    print("\n")        
        
# --------------------------------------------------------------------

if __name__ == "__main__":

    # Task 1 - make_graph() with return value (graph).
    print("-------------------------------------------------------------------------------")
    print("                                    TASK1                                      ") 
    print("-------------------------------------------------------------------------------")

    graph = make_graph()
    # Help function for printing all vertex and their edges in graph.
    print_graph(graph)

    
    # Task 2 - shortest_path(graph, node) with return value (List, int).
    print("-------------------------------------------------------------------------------")
    print("                                    TASK2                                      ") 
    print("-------------------------------------------------------------------------------")
    
    print("\nShortest path to", graph[3].d1)
    shortest_path_list, w = shortest_path(graph, graph[3])
    # Help function for printing shortest path with step at the moment and full weight at the end!
    print_shortest_path(shortest_path_list, w)

    print("\nShortest path to", graph[5].d1)
    shortest_path_list, w = shortest_path(graph, graph[5])
    # Help function for printing shortest path with step at the moment and full weight at the end!
    print_shortest_path(shortest_path_list, w)

    print("\nShortest path to", graph[6].d1)
    shortest_path_list, w = shortest_path(graph, graph[6])
    # Help function for printing shortest path with step at the moment and full weight at the end!
    print_shortest_path(shortest_path_list, w)
    
    print("\nShortest path to", graph[7].d1)
    shortest_path_list, w = shortest_path(graph, graph[7])
    # Help function for printing shortest path with step at the moment and full weight at the end!
    print_shortest_path(shortest_path_list, w)

    print("\nShortest path to", graph[8].d1)
    shortest_path_list, w = shortest_path(graph, graph[8])
    # Help function for printing shortest path with step at the moment and full weight at the end!
    print_shortest_path(shortest_path_list, w)
    
    # Task3 - BFS(graph, start_node) with return value (Dict<string, int>)
    print("-------------------------------------------------------------------------------")
    print("                                    TASK3                                      ") 
    print("-------------------------------------------------------------------------------")

    d = BFS(graph, graph[0])
    # Result of BFS printed as a dictionaty.
    print("\n", d, "\n")
    
    # Task4 - BFS_tree(graph, start_node) with return value (tree)
    print("-------------------------------------------------------------------------------")
    print("                                    TASK4                                      ") 
    print("-------------------------------------------------------------------------------")
    t = BFS_tree(graph, graph[0])
    # Result of BFS_tree printed as Nodes with their parents.
    print_tree(t)    

  
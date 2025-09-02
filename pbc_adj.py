import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

def make_initial_circle(G, m):
    """
    Create an initial cycle of m nodes in the graph G.
    
    Args:
        G (networkx.Graph): Graph object.
        m (int): Number of nodes in the initial cycle.
    
    Returns:
        list: Nodes belonging to the initial cycle.
    """
    nodes = list(range(m))             # [0, 1, ..., m-1]
    G.add_nodes_from(nodes)

    # connect in a cycle
    for i in range(m):
        G.add_edge(nodes[i], nodes[(i + 1) % m])

    return nodes

def build_hyperbolic_lattice(Nc, k, m, seed=2):
    """
    Build adjacency matrix of a {m,k} hyperbolic lattice patch with Nc circles.
    
    Args:
        Nc (int): number of concentric circles (>=2).
        k (int): target degree for each node.
        m (int): number of nodes in the initial polygon (>=3).
        seed (int): random seed for reproducibility.

    Returns:
        H (numpy.ndarray): adjacency matrix of the lattice.
        G (networkx.Graph): graph object.
        circles (list): list of circles, each a list of node indices.
    """
    random.seed(seed)
    np.random.seed(seed)

    # initialize graph
    G = nx.Graph()

    # first circle: m-gon
    circles = []
    first_circle = make_initial_circle(G, m)
    circles.append(first_circle)
    node_count = m

    # add further circles
    for cd in range(2, Nc + 1):
        prev_circle = circles[-1]
        new_circle = []

        for parent in prev_circle:
            # while parent has degree < k, add new neighbors
            while G.degree(parent) < k:
                new_node = node_count
                node_count += 1

                # connect parent -> new node
                G.add_node(new_node)
                G.add_edge(parent, new_node)
                new_circle.append(new_node)

                # also connect consecutive new nodes to form a ring
                if len(new_circle) > 1:
                    G.add_edge(new_circle[-2], new_circle[-1])

        # close the ring
        if len(new_circle) > 1:
            G.add_edge(new_circle[0], new_circle[-1])

        circles.append(new_circle)

    # Fix nodes with degree < k by pairing them
    underfull = [n for n in G.nodes if G.degree(n) < k]
    while len(underfull) > 1:
        u = random.choice(underfull)
        # underfull.remove(u)

        # pick another underfull node not already connected
        #candidates = [n for n in underfull if not G.has_edge(u, n)]
        candidates = [n for n in G.nodes]
        if not candidates:
            continue
        v = random.choice(candidates)
        # underfull.remove(v)

        G.add_edge(u, v)

        # refresh list
        underfull = [n for n in G.nodes if G.degree(n) < k]

    # Safety check: ensure all nodes have degree == k
    # for n in G.nodes:
    #     if G.degree(n) != k:
    #         raise ValueError(f"Node {n} has degree {G.degree(n)}, expected {k}")

    # Convert to adjacency matrix
    H = nx.to_numpy_array(G, dtype=int)

    return H, G, circles

def print_row_sums(H):
    """
    Print the sum of each row of the adjacency matrix.
    
    Args:
        H (numpy.ndarray): adjacency matrix.
    """
    row_sums = H.sum(axis=1)
    for i, s in enumerate(row_sums):
        print(f"Row {i}: {s}")

def visualize_lattice(G, circles):
    """
    Visualize the lattice using concentric circles for each layer.
    
    Args:
        G (networkx.Graph): graph object.
        circles (list): list of circles, each a list of node indices.
    """
    pos = {}
    radius_step = 2.0  # spacing between concentric circles

    for layer, circle in enumerate(circles):
        radius = (layer + 1) * radius_step
        angle_step = 2 * np.pi / len(circle)
        for i, node in enumerate(circle):
            angle = i * angle_step
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

    plt.figure(figsize=(8, 8))
    nx.draw(
        G, pos, with_labels=True,
        node_size=300, node_color="lightblue", edge_color="gray"
    )
    plt.axis("equal")
    plt.show()

#H, G, circles = build_hyperbolic_lattice(Nc=4, k=8, m=3, seed=1) #m == p, k == q
#print_row_sums(H)
#visualize_lattice(G, circles)

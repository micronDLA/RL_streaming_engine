import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_se_structure(GRID_SIZE = 4):
    """Generates 1x16 structure for the streaming engine

    Args:
        GRID_SIZE (int, optional): Dimension of the grid. Value 4 generates 16 tiles, 
        interconnected acc. to 1x16 architecture. Defaults to 4.

    Returns:
        G: Networkx graph for streaming engine
    """    
    G = nx.Graph()
    node_list = [(i,j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
    print('Node list:', node_list, end='\n\n')
    G.add_nodes_from(node_list)

    print('Generate Connections (invalid node ids not added):')
    # Loop over nodes to add edges with hops as weights
    for x_idx, y_idx in node_list:
        # connect to Up
        if y_idx + 1 < GRID_SIZE:
            x_up_idx, y_up_idx = x_idx, y_idx + 1
        else:
            x_up_idx, y_up_idx = x_idx + 1, 0

        if (x_up_idx, y_up_idx) in node_list:
            G.add_edge((x_idx, y_idx), (x_up_idx, y_up_idx), weight=1)

        # connect to Down
        if y_idx - 1 >= 0:
            x_down_idx, y_down_idx = x_idx, y_idx - 1
        else:
            x_down_idx, y_down_idx = x_idx - 1, GRID_SIZE - 1

        if (x_down_idx, y_down_idx) in node_list:
            G.add_edge((x_idx, y_idx), (x_down_idx, y_down_idx), weight=1)
        
        # connect to Up2
        if y_idx + 2 < GRID_SIZE:
            x_up2_idx, y_up2_idx = x_idx, y_idx + 2
        else:
            x_up2_idx, y_up2_idx = x_idx + 1, (y_idx + 2) % GRID_SIZE

        if (x_up2_idx, y_up2_idx) in node_list:
            G.add_edge((x_idx, y_idx), (x_up2_idx, y_up2_idx), weight=1)

        # connect to Down2
        if y_idx - 2 >= 0:
            x_down2_idx, y_down2_idx = x_idx, y_idx - 2
        else:
            x_down2_idx, y_down2_idx = x_idx - 1, (y_idx - 2) % GRID_SIZE
            
        if (x_down2_idx, y_down2_idx) in node_list:
            G.add_edge((x_idx, y_idx), (x_down2_idx, y_down2_idx), weight=1)

        print(f'Tile: {(x_idx, y_idx)} | Up: {(x_up_idx, y_up_idx)} | Down: {(x_down_idx, y_down_idx)} | Up2: {(x_up2_idx, y_up2_idx)} | Down2: {(x_down2_idx, y_down2_idx)}')

    print()
    # Generate adjacency matrix to verify connections 
    A = nx.to_numpy_matrix(G, nodelist=node_list)
    print('Adjacency matrix:')
    print(A)
    
    return G
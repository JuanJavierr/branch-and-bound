import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

node_counter = 0
edges = []
best_solution = ((None, None), float('-inf'))  # (solution, objective value)
node_values = {}


def branch_and_bound(x1_range=(0, None), x2_range=(0, None), parent_constraint=None):
    global node_counter, best_solution, edges, node_values

    c = [-1, -2]
    A_ub = [
        [4, 0],
        [0, 1],
        [2, 3],
        [-10, 15],
        [-20, -15]
    ]
    b_ub = [13, 3, 14, 32, -16]

    bounds = [x1_range, x2_range]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    node_id = node_counter
    node_counter += 1

    if res.success:
        x1_val, x2_val = res.x
        objective = -res.fun  # Since we are minimizing the negative of the objective function

        # Update node_values with the current node's x, y values, and objective value
        node_values[node_id] = (round(x1_val, 2), round(x2_val, 2), round(objective, 2), node_id, parent_constraint)

        # Update the best solution if a new better integer solution is found
        if x1_val.is_integer() and x2_val.is_integer() and objective > best_solution[1]:
            best_solution = ((x1_val, x2_val), objective)

        # Branch on x1 if not integer
        if not x1_val.is_integer():
            x1_floor = int(np.floor(x1_val))
            edges.append((node_id, node_counter, f"x <= {x1_floor}"))
            branch_and_bound(x1_range=(0, x1_floor), x2_range=x2_range, parent_constraint=f"x <= {x1_floor}")
            edges.append((node_id, node_counter, f"x >= {x1_floor + 1}"))
            branch_and_bound(x1_range=(x1_floor+1, None), x2_range=x2_range, parent_constraint=f"x >= {x1_floor + 1}")

        # Branch on x2 if not integer
        if not x2_val.is_integer():
            x2_floor = int(np.floor(x2_val))
            edges.append((node_id, node_counter, f"y <= {x2_floor}"))
            branch_and_bound(x1_range=x1_range, x2_range=(0, x2_floor), parent_constraint=f"y <= {x2_floor}")
            edges.append((node_id, node_counter, f"y >= {x2_floor + 1}"))
            branch_and_bound(x1_range=x1_range, x2_range=(x2_floor+1, None), parent_constraint=f"y >= {x2_floor + 1}")
    else:
        # Update node_values with the current node's x, y values, and objective value
        node_values[node_id] = (None, None, None, node_id, parent_constraint)


def visualize_tree(edges, node_values):
    G = nx.DiGraph()
    G.add_edges_from([(u, v) for u, v, _ in edges])
    
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Draw nodes (invisible)
    nx.draw_networkx_nodes(G, pos, node_color='white', node_size=1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Draw labels with a bounding box around each
    labels = {}
    for node, values in node_values.items():
        if values[0] is None:
            labels[node] = f"id={values[3]}, Infeasible"
        else:
            labels[node] = f"id={values[3]}, x={values[0]}, y={values[1]}, z={values[2]}"
        
    nx.draw_networkx_labels(G, pos, labels, bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    
    # Draw edge labels for constraints
    edge_labels = {(u, v): constraint for u, v, constraint in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()


# Execute the Branch-and-Bound algorithm and then print and visualize the result
branch_and_bound()

print(f"Best integer result: x={best_solution[0][0]}, y={best_solution[0][1]}")

# Visualize the tree
visualize_tree(edges, node_values)

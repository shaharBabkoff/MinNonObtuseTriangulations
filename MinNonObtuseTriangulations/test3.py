import math
from fractions import Fraction
from pathlib import Path
from cgshop2025_pyutils import (
    DelaunayBasedSolver,
    InstanceDatabase,
    ZipSolutionIterator,
    ZipWriter,
    verify,
)
from colorama import Fore, Style
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from our_solution3 import OurSolution3


# Function to visualize the instance
def visualize_instance(instance):
    G = nx.Graph()

    # Add nodes
    for idx, (x, y) in enumerate(zip(instance.points_x, instance.points_y)):
        G.add_node(idx, pos=(x, y))  # Position of the node

    # Add edges from boundaries
    for i in range(len(instance.region_boundary) - 1):
        G.add_edge(instance.region_boundary[i], instance.region_boundary[i + 1])
    G.add_edge(instance.region_boundary[-1], instance.region_boundary[0])  # Close the polygon

    # Add edges from additional constraints
    for constraint in instance.additional_constraints:
        G.add_edge(constraint[0], constraint[1])

    # Get positions and plot
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, edge_color='gray')
    plt.title(f"Visualization of Instance: {instance.instance_uid}")
    plt.show()


# Function to visualize the solution with Steiner points
def visualize_solution(instance, solution, steiner_points):
    points = [(x, y) for x, y in zip(instance.points_x, instance.points_y)]
    points.extend([(float(Fraction(x)), float(Fraction(y))) for x, y in steiner_points])  # Ensure proper float conversion

    edges = solution.edges
    triangles = validate_edges_and_triangles(edges, points)  # Validate triangles

    fig, ax = plt.subplots()
    ax.set_title(f"Solution Visualization: {solution.instance_uid}")

    # Plot original points
    for idx, (x, y) in enumerate(points[:len(instance.points_x)]):
        ax.plot(x, y, "o", color="blue", label=f"Point {idx}" if idx == 0 else "")

    # Plot Steiner points
    for x, y in steiner_points:
        ax.plot(float(Fraction(x)), float(Fraction(y)), "o", color="green", markersize=8, label="Steiner Point")

    # Plot edges
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=1)

    # Plot triangles
    for triangle in triangles:
        t_points = [points[idx] for idx in triangle]
        polygon = Polygon(t_points, closed=True, edgecolor="blue", facecolor="none", linewidth=1)
        ax.add_patch(polygon)

    plt.legend(["Original Points", "Steiner Points"])
    plt.show()


def validate_edges_and_triangles(edges, points):
    """
    Validate edges and extract triangles from them.
    """
    triangles = set()
    for edge1 in edges:
        for edge2 in edges:
            if edge1 != edge2:
                for edge3 in edges:
                    if edge3 != edge1 and edge3 != edge2:
                        # Check if they form a valid triangle
                        vertices = {edge1[0], edge1[1], edge2[0], edge2[1], edge3[0], edge3[1]}
                        if len(vertices) == 3:  # Only three unique vertices form a triangle
                            triangles.add(tuple(sorted(vertices)))
    return triangles

# Step 1: Load instances
instance_path = "example.zip"
idb = InstanceDatabase(instance_path)

# Step 2: Delete existing solution ZIP if present
if Path("example_sol2.zip").exists():
    Path("example_sol2.zip").unlink()

# Step 3: Solve instances
solutions = []
for instance in idb:
    print(f"Processing Instance: {instance.instance_uid}")

    # Visualize instance before solving
    visualize_instance(instance)

    # Solve the problem
    solver = OurSolution3(instance)
    solution = solver.ourThirdSol()
    solutions.append(solution)

    # Visualize the solution with triangles
    steiner_points = list(zip(solution.steiner_points_x, solution.steiner_points_y))
    visualize_solution(instance, solution, steiner_points)

# Step 4: Write solutions to ZIP
with ZipWriter("example_sol2.zip") as zw:
    for solution in solutions:
        zw.add_solution(solution)

# Step 5: Verify solutions
for solution in ZipSolutionIterator("example_sol2.zip"):
    instance = idb[solution.instance_uid]
    result = verify(instance, solution)

    # Print verification results
    print(f"Verification Results for {solution.instance_uid}:")
    print(result)
    print(f"Number of obtuse triangles: {result.num_obtuse_triangles}")
    if result.num_obtuse_triangles > 0:
        print(Fore.BLUE + "Obtuse triangles remain in the solution!" + Style.RESET_ALL)
    else:
        print(Fore.GREEN + "Solution is valid with no obtuse triangles!" + Style.RESET_ALL)

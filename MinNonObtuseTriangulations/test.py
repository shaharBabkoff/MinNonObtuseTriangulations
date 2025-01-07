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

from our_solution import OurSolution


# פונקציה ליצירת ויזואליזציה של המקרה
def visualize_instance(instance):
    G = nx.Graph()

    # הוספת צמתים
    for idx, (x, y) in enumerate(zip(instance.points_x, instance.points_y)):
        G.add_node(idx, pos=(x, y))  # מיקום הצומת

    # הוספת הקשתות מהגבולות
    for i in range(len(instance.region_boundary) - 1):
        G.add_edge(instance.region_boundary[i], instance.region_boundary[i + 1])
    G.add_edge(instance.region_boundary[-1], instance.region_boundary[0])  # סגירת המצולע

    # הוספת קשתות ממגבלות נוספות
    for constraint in instance.additional_constraints:
        G.add_edge(constraint[0], constraint[1])

    # קבלת המיקומים והשרטוט
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, edge_color='gray')
    plt.title(f"Visualization of Instance: {instance.instance_uid}")
    plt.show()


# פונקציה להצגת המשולשים של הפתרון
def visualize_solution(instance, solution, steiner_points):
    points = [(x, y) for x, y in zip(instance.points_x, instance.points_y)]
    points.extend(steiner_points)  # הוספת נקודות שטיינר לרשימת הנקודות
    patches_obtuse = []  # משולשים עם זווית קהה
    patches_acute = []   # משולשים חדים
    steiner_x, steiner_y = zip(*steiner_points) if steiner_points else ([], [])

    # בדיקה וחלוקה של משולשים
    for edge1, edge2 in solution.edges:
        if edge1 < len(points) and edge2 < len(points):
            p1, p2 = points[edge1], points[edge2]
            patches_acute.append(Polygon([p1, p2], closed=False))

    # שרטוט
    fig, ax = plt.subplots()
    ax.set_title(f"Solution Visualization: {solution.instance_uid}")

    # ציור הנקודות
    for idx, (x, y) in enumerate(points[:len(instance.points_x)]):
        ax.plot(x, y, "o", color="blue", label=f"Point {idx}" if idx == 0 else "")

    # ציור נקודות שטיינר
    for x, y in steiner_points:
        ax.plot(x, y, "o", color="green", markersize=8, label="Steiner Point")

    # ציור משולשים - משולשים חדים (בכחול) ומשולשים קהים (באדום)
    p_obtuse = PatchCollection(patches_obtuse, edgecolor="red", facecolor="none", linewidths=1)
    p_acute = PatchCollection(patches_acute, edgecolor="blue", facecolor="none", linewidths=1)

    ax.add_collection(p_acute)
    ax.add_collection(p_obtuse)
    plt.legend(["Original Points", "Steiner Points"])
    plt.show()


# שלב 1: טעינת המקרים
instance_path = "example2.zip"
idb = InstanceDatabase(instance_path)

# שלב 2: מחיקת קובץ ZIP קודם אם קיים
if Path("example_sol.zip").exists():
    Path("example_sol.zip").unlink()

# שלב 3: יצירת פתרונות באמצעות פתרון נאיבי
solutions = []
for instance in idb:
    print(f"Processing Instance: {instance.instance_uid}")

    # ויזואליזציה של המקרה לפני הפתרון
    visualize_instance(instance)

    # פתרון הבעיה
    solver = OurSolution(instance)  # פתרון נאיבי
    solution = solver.ourFirstSol()
    solutions.append(solution)

    # ויזואליזציה של הפתרון עם המשולשים
    steiner_points = list(zip(solution.steiner_points_x, solution.steiner_points_y))
    visualize_solution(instance, solution, steiner_points)

# שלב 4: כתיבת הפתרונות לקובץ ZIP
with ZipWriter("example_sol.zip") as zw:
    for solution in solutions:
        zw.add_solution(solution)

# שלב 5: אימות הפתרונות
for solution in ZipSolutionIterator("example_sol.zip"):
    instance = idb[solution.instance_uid]
    result = verify(instance, solution)
    if result.num_obtuse_triangles > 0:
        print(Fore.BLUE + "existing obtuse triangles" + Style.RESET_ALL)
    print(f"{solution.instance_uid}: {result}")
    assert not result.errors, "הפתרון מכיל שגיאות!"

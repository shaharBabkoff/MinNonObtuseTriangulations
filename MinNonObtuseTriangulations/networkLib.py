import networkx as nx
import matplotlib.pyplot as plt

# יצירת גרף
G = nx.Graph()

# הוספת צמתים וקשתות
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# שרטוט הגרף
nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
plt.show()

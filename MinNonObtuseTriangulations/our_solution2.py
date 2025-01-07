import math
from itertools import combinations
from shapely.geometry import Point as ShapelyPoint, LineString
from fractions import Fraction

from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance
from cgshop2025_pyutils.data_schemas.solution import Cgshop2025Solution
from cgshop2025_pyutils.geometry import Point, ConstrainedTriangulation


class OurSolution2:
    def __init__(self, instance: Cgshop2025Instance):
        self.instance = instance
        self.steiner_points = []  # רשימת נקודות שטיינר
        self.steiner_points_x = []
        self.steiner_points_y = []
        self.edges_added = []  # רשימה לצלעות שנוספו
        self.obtuse_triangles = []  # List to store obtuse triangles

    def calculate_angle(self,a, b, c):
        ab = (b[0] - a[0], b[1] - a[1])  # Vector from a to b
        cb = (b[0] - c[0], b[1] - c[1])  # Vector from c to b

        dot_product = ab[0] * cb[0] + ab[1] * cb[1]
        magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
        magnitude_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

        if magnitude_ab < 1e-9 or magnitude_cb < 1e-9:
            print(f"Warning: Zero magnitude vector detected. Points: a={a}, b={b}, c={c}")
            return 0.0

        cos_theta = dot_product / (magnitude_ab * magnitude_cb)
        cos_theta = max(-1, min(1, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def validate_points(self, p1, p2, p3, threshold=1e-6):
        def distance(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        return (
                distance(p1, p2) > threshold and
                distance(p2, p3) > threshold and
                distance(p3, p1) > threshold
        )

    def perpendicular_projection(self, v, p1, p2):
        """
        Finds the perpendicular projection of v onto the segment p1-p2.
        Returns the projection as rational numbers (fractions).
        """
        vx, vy = Fraction(v[0]), Fraction(v[1])
        x1, y1 = Fraction(p1[0]), Fraction(p1[1])
        x2, y2 = Fraction(p2[0]), Fraction(p2[1])

        # Handle special cases for vertical and horizontal segments
        if x1 == x2:  # Vertical segment
            proj_y = max(min(vy, max(y1, y2)), min(y1, y2))
            return x1.limit_denominator(), proj_y.limit_denominator()
        elif y1 == y2:  # Horizontal segment
            proj_x = max(min(vx, max(x1, x2)), min(x1, x2))
            return proj_x.limit_denominator(), y1.limit_denominator()

        # Compute slope of the segment
        m = (y2 - y1) / (x2 - x1)

        # Line equation: y = mx + c -> c = y1 - m*x1
        c = y1 - m * x1

        # Perpendicular slope
        m_perpendicular = -1 / m

        # Line 2 (perpendicular): y = m_perpendicular * x + b
        b_perpendicular = vy - m_perpendicular * vx

        # Solve for x (intersection of lines)
        proj_x = (b_perpendicular - c) / (m - m_perpendicular)

        # Solve for y
        proj_y = m * proj_x + c

        # Clamp the projection to segment bounds
        proj_x = max(min(proj_x, max(x1, x2)), min(x1, x2))
        proj_y = max(min(proj_y, max(y1, y2)), min(y1, y2))

        return proj_x.limit_denominator(), proj_y.limit_denominator()

    def ourSecondSol(self) -> Cgshop2025Solution:
        ct = ConstrainedTriangulation()
        instance = self.instance
        for x, y in zip(instance.points_x, instance.points_y):
            ct.add_point(Point(x, y))
        ct.add_boundary(instance.region_boundary)
        for constraint in instance.additional_constraints:
            ct.add_segment(constraint[0], constraint[1])
        edges = ct.get_triangulation_edges()

        triangles = set()
        for a, b, c in combinations(range(len(instance.points_x)), 3):
            if ({a, b} in map(set, edges) and {b, c} in map(set, edges) and {c, a} in map(set, edges)):
                triangles.add(tuple(sorted([a, b, c])))

        points = list(zip(instance.points_x, instance.points_y))
        MAX_STEINER_POINTS = 1000
        self.steiner_points_x = []
        self.steiner_points_y = []
        self.edges_added = []

        while triangles:
            t = triangles.pop()
            p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]

            if not self.validate_points(p1, p2, p3):
                print(f"Skipping degenerate triangle with points: {p1}, {p2}, {p3}")
                continue

            angles = [
                (self.calculate_angle(p1, p2, p3), p1, p2, p3),  # Angle at p2
                (self.calculate_angle(p2, p3, p1), p2, p3, p1),  # Angle at p3
                (self.calculate_angle(p3, p1, p2), p3, p1, p2)  # Angle at p1
            ]
            print(f"angles: {angles}")

            for angle, p1, vertex, p2 in angles:
                if angle > 90:
                    print(f"Found obtuse triangle: {angle} at vertex {vertex}")
                    print(f"p1: {p1}, p2: {p2}, vertex: {vertex}")

                    # Remove the original edge
                    original_edge = (points.index(p1), points.index(p2))
                    if original_edge in edges:
                        edges.remove(original_edge)
                    elif original_edge[::-1] in edges:
                        edges.remove(original_edge[::-1])

                    if len(self.steiner_points_x) >= MAX_STEINER_POINTS:
                        print("Reached maximum number of Steiner points. Stopping.")
                        break

                    # Calculate the perpendicular point
                    steiner_point = self.perpendicular_projection(vertex, p1, p2)

                    # Verify the angle at the Steiner point
                    perpendicular_angle = self.calculate_angle(vertex,
                                                               (float(steiner_point[0]), float(steiner_point[1])), p1)
                    print(f"Perpendicular point: {steiner_point}")
                    print(f"Angle at perpendicular point: {perpendicular_angle:.2f} degrees")
                    if abs(perpendicular_angle - 90) > 1e-5:
                        print(f"Error: Angle at Steiner Point {steiner_point} is not 90 degrees!")

                    # Add Steiner point as rationalized coordinates
                    self.steiner_points.append((str(steiner_point[0]), str(steiner_point[1])))
                    self.steiner_points_x.append(str(steiner_point[0]))
                    self.steiner_points_y.append(str(steiner_point[1]))
                    print(f"Added Steiner Point: {steiner_point}")

                    # Add new edges and retain the original edge if necessary
                    points.append((float(steiner_point[0]), float(steiner_point[1])))
                    new_index = len(points) - 1

                    # Check if the original edge (p1, p2) is critical for other triangles
                    if (points.index(p1), points.index(p2)) in edges or (points.index(p2), points.index(p1)) in edges:
                        self.edges_added.append((points.index(p1), points.index(p2)))

                    self.edges_added.extend([
                        (points.index(p1), new_index),
                        (new_index, points.index(p2)),
                        (new_index, points.index(vertex)),
                    ])

                    # Rebuild the list of triangles
                    triangles.clear()
                    for a, b, c in combinations(range(len(points)), 3):
                        if ({a, b} in map(set, edges) and {b, c} in map(set, edges) and {c, a} in map(set, edges)):
                            triangles.add(tuple(sorted([a, b, c])))

        edges += self.edges_added
        # edges+= (1,6), (9,6),(1,9)

        print("Final Steiner Points:", self.steiner_points)
        print("Edges Added:", self.edges_added)
        print("total edges", edges)
        print("total points", points)
        # Calculate total triangles including new ones
        all_triangles = set()
        for a, b, c in combinations(range(len(points)), 3):
            if ({a, b} in map(set, edges) and {b, c} in map(set, edges) and {c, a} in map(set, edges)):
                all_triangles.add(tuple(sorted([a, b, c])))

        print("Total triangles including new ones:", len(all_triangles))
        print("Triangles and their points:")
        for triangle in all_triangles:
            p1, p2, p3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
            print(f"Triangle: {triangle}, Points: {p1}, {p2}, {p3}")

        return Cgshop2025Solution(
            content_type="CG_SHOP_2025_Solution",
            instance_uid=instance.instance_uid,
            steiner_points_x=self.steiner_points_x,
            steiner_points_y=self.steiner_points_y,
            edges=edges,
        )
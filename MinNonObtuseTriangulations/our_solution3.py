import math
import numpy as np
from sympy import symbols, Eq, solve
from itertools import combinations
from shapely.geometry import Point as ShapelyPoint, LineString
from fractions import Fraction

from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance
from cgshop2025_pyutils.data_schemas.solution import Cgshop2025Solution
from cgshop2025_pyutils.geometry import Point, ConstrainedTriangulation

class OurSolution3:
    def __init__(self, instance: Cgshop2025Instance):
        self.instance = instance
        self.steiner_points = []  # List of Steiner points
        self.steiner_points_x = []
        self.steiner_points_y = []
        self.edges_added = []  # List of edges added
        self.obtuse_triangles = []  # List to store obtuse triangles


    # A function to check whether point P(x, y)
    # lies inside the triangle formed by
    # A(x1, y1), B(x2, y2) and C(x3, y3)
    def get_triangle_edges(self,triangle):
        return [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0])
        ]
    def calculate_line(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        if x1 == x2:
            return None, x1  # Vertical line, return x-intercept only

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def isInsideTriangle(self, triangle, x, y):
        """
        Check if a given point lies on any edge of a triangle.

        Parameters:
        triangle (list of tuples): List of three vertices defining the triangle, e.g., [(x1, y1), (x2, y2), (x3, y3)].
        x, y (float): The coordinates of the point to check.

        Returns:
        bool: True if the point lies on any edge of the triangle, False otherwise.
        """

        def is_point_on_line_segment(x1, y1, x2, y2, x, y):
            """Check if point (x, y) is on the line segment defined by points (x1, y1) and (x2, y2)."""
            # Check if the area formed by the points is zero (collinear)
            area = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
            if area != 0:
                return False

            # Check if the point is within the segment bounds
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)

            return min_x <= x <= max_x and min_y <= y <= max_y

        # Unpack triangle vertices
        v1, v2, v3 = triangle

        # Check if the point lies on any of the three edges
        return (
                is_point_on_line_segment(v1[0], v1[1], v2[0], v2[1], x, y) or
                is_point_on_line_segment(v2[0], v2[1], v3[0], v3[1], x, y) or
                is_point_on_line_segment(v3[0], v3[1], v1[0], v1[1], x, y)
        )

    def find_triangles_with_edge_excluding(self, edge, triangles, current_triangle):
        edge_set = set(edge)
        return [triangle for triangle in triangles if edge_set.issubset(set(triangle)) and triangle != current_triangle]

    def validate_intersection_and_triangle(points, intersection_point, triangles, edges):
        """
        Validate the intersection point and ensure proper triangle closure.

        Parameters:
            points (list): List of existing points.
            intersection_point (tuple): The intersection point to validate.
            triangles (set): Set of current triangles.
            edges (list): List of current edges.

        Returns:
            bool: True if the intersection and resulting triangles are valid, False otherwise.
        """
        # Check if the intersection point is already in the points list
        if intersection_point in points:
            print(f"Intersection point {intersection_point} already exists in points.")
            return False

        # Check if the intersection point forms valid triangles with existing points
        for edge in edges:
            p1, p2 = edge
            new_triangle = tuple(sorted([p1, p2, len(points)]))  # Form a new triangle with the intersection point
            if new_triangle in triangles:
                print(f"Triangle {new_triangle} already exists.")
                return False

        print(f"Intersection point {intersection_point} validated and forms valid triangles.")
        return True

    def find_other_edges_equations(self, edge, triangle, points):
        other_vertices = [v for v in triangle if v not in edge]
        if len(other_vertices) != 1:
            raise ValueError(f"Invalid triangle or edge: {triangle}, {edge}")

        other_vertex = other_vertices[0]
        edge1 = (other_vertex, edge[0])
        edge2 = (other_vertex, edge[1])

        line1_eq, m1, c1 = self.find_line_equation(points[edge1[0]], points[edge1[1]])
        line2_eq, m2, c2 = self.find_line_equation(points[edge2[0]], points[edge2[1]])

        print(f"Line equations for edges in triangle {triangle}: {line1_eq}, {line2_eq}")
        return [(line1_eq, m1, c1), (line2_eq, m2, c2)]

    def find_line_equation(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        if x1 == x2:
            return f"x = {x1}", None, None

        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        return f"y = {m}x + {c}", m, c

    def find_intersection(self, line1, line2):
        m1, c1 = line1
        m2, c2 = line2

        if m1 == m2:
            if c1 == c2:
                return None  # Identical lines
            else:
                return None  # Parallel lines

        x, y = symbols('x y')
        eq1 = Eq(y, m1 * x + c1)
        eq2 = Eq(y, m2 * x + c2)
        solution = solve((eq1, eq2), (x, y))
        print(f"Intersection between {line1} and {line2}: {solution}")
        return solution

    def calculate_angle(self, a, b, c):
        ab = (b[0] - a[0], b[1] - a[1])
        cb = (b[0] - c[0], b[1] - c[1])

        dot_product = ab[0] * cb[0] + ab[1] * cb[1]
        magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
        magnitude_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

        if magnitude_ab < 1e-9 or magnitude_cb < 1e-9:
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
        vx, vy = Fraction(v[0]), Fraction(v[1])
        x1, y1 = Fraction(p1[0]), Fraction(p1[1])
        x2, y2 = Fraction(p2[0]), Fraction(p2[1])

        if x1 == x2:
            proj_y = max(min(vy, max(y1, y2)), min(y1, y2))
            return x1.limit_denominator(), proj_y.limit_denominator()
        elif y1 == y2:
            proj_x = max(min(vx, max(x1, x2)), min(x1, x2))
            return proj_x.limit_denominator(), y1.limit_denominator()

        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        m_perpendicular = -1 / m
        b_perpendicular = vy - m_perpendicular * vx

        proj_x = (b_perpendicular - c) / (m - m_perpendicular)
        proj_y = m * proj_x + c

        proj_x = max(min(proj_x, max(x1, x2)), min(x1, x2))
        proj_y = max(min(proj_y, max(y1, y2)), min(y1, y2))

        print(f"Perpendicular projection of {v} on line {p1}-{p2}: ({proj_x}, {proj_y})")
        return proj_x.limit_denominator(), proj_y.limit_denominator()

    def ourThirdSol(self) -> Cgshop2025Solution:
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


        print(f"Initial Triangles: {triangles}")
        print(f"Initial Points: {points}")

        while triangles:
            t = triangles.pop()
            p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]

            print(f"Processing Triangle: {t} -> Points: {p1}, {p2}, {p3}")

            if not self.validate_points(p1, p2, p3):
                print(f"Skipping degenerate triangle: {t}")
                continue

            angles = [
                (self.calculate_angle(p1, p2, p3), p1, p2, p3),
                (self.calculate_angle(p2, p3, p1), p2, p3, p1),
                (self.calculate_angle(p3, p1, p2), p3, p1, p2)
            ]
            print(f"angles: {angles}")
            for angle, p1, vertex, p2 in angles:
                print(f"Checking angle at vertex {vertex}: {angle}°")
                if angle > 90:
                    print(f"Obtuse angle detected: {angle}° at vertex {vertex}")
                    # Remove the original edge
                    original_edge = (points.index(p1), points.index(p2))
                    if original_edge in edges:
                        edges.remove(original_edge)
                    elif original_edge[::-1] in edges:
                        edges.remove(original_edge[::-1])
                    if len(self.steiner_points_x) >= MAX_STEINER_POINTS:
                        print("Reached MAX_STEINER_POINTS. Stopping further processing.")
                        break

                    steiner_point = self.perpendicular_projection(vertex, p1, p2)
                    proj_x = steiner_point[0]
                    proj_y = steiner_point[1]
                    # Verify the angle at the Steiner point
                    perpendicular_angle = self.calculate_angle(vertex,
                                                               (float(steiner_point[0]), float(steiner_point[1])), p1)
                    print(f"Perpendicular point: {steiner_point}")
                    print(f"Angle at perpendicular point: {perpendicular_angle:.2f} degrees")
                    if abs(perpendicular_angle - 90) > 1e-5:
                        print(f"Error: Angle at Steiner Point {steiner_point} is not 90 degrees!")



                    self.steiner_points.append((str(proj_x), str(proj_y)))
                    self.steiner_points_x.append(str(proj_x))
                    self.steiner_points_y.append(str(proj_y))
                    print(f"Added Steiner Point: {steiner_point}")
                    points.append((proj_x, proj_y))
                    new_index = len(points) - 1

                    print(f"Added Steiner Point: {steiner_point}. Updating edges.")
                    self.edges_added.append((points.index(vertex), new_index))
                    print(f"Added edge: {points.index(vertex)} -> {new_index}")
                    edge_excluding=points.index(p1), points.index(p2)
                    triangles_with_edge = self.find_triangles_with_edge_excluding(
                        edge_excluding, triangles, t
                    )

                    if triangles_with_edge:
                        print(
                            f"Found adjacent triangles sharing edge {(points.index(p1), points.index(p2))}: {triangles_with_edge}")
                        for triangle in triangles_with_edge:
                            # All edges of the triangle
                            # Exclude the edge_excluding and calculate other edges
                            edges = self.get_triangle_edges(triangle)
                            other_edges = [edge for edge in edges if set(edge) != set(edge_excluding)]
                            # Compute line equations for the other two edges
                            lines = [self.calculate_line(points[edge[0]], points[edge[1]]) for edge in other_edges]
                            m_steiner,b_steiner = self.calculate_line(vertex,steiner_point)
                            p1, p2, p3 = [points[vertex] for vertex in triangle]

                            # Print the triangle points for debugging
                            print(f"Checking Triangle: {triangle}, Points: {p1}, {p2}, {p3}")
                                # Perform necessary operations if the point is inside
                            # Calculate line equations for each edge of the triangl

                            for line in lines:
                                m, b = line
                                if m is not None:
                                    intersection = self.find_intersection((m, b), (m_steiner, b_steiner))
                                    if intersection:
                                        intersection_point = (
                                            Fraction(str(intersection[symbols('x')])),
                                            Fraction(str(intersection[symbols('y')]))
                                        )
                                        print(f"Found Intersection Point: {intersection_point}")

                                        # Check if the point is inside the triangle
                                        triangle_coords = [points[vertex] for vertex in triangle]
                                        if self.isInsideTriangle(triangle_coords, intersection_point[0],
                                                                 intersection_point[1]):
                                            print(f"Point {intersection_point} is inside the triangle {triangle}.")
                                            self.steiner_points.append(
                                                (str(intersection_point[0]), str(intersection_point[1]))
                                            )
                                            self.steiner_points_x.append(str(intersection_point[0]))
                                            self.steiner_points_y.append(str(intersection_point[1]))
                                            points.append(intersection_point)
                                            new_intersection_index = len(points) - 1
                                            self.edges_added.append((new_index, new_intersection_index))
                                            print(
                                                f"Added new intersection edge: {new_index} -> {new_intersection_index}")
                                        else:
                                            print(
                                                f"Intersection point {intersection_point} is outside the triangle. Skipping.")

                    self.edges_added.append((new_index, points.index(p2)))

        edges += self.edges_added

        self.steiner_points_x = [str(Fraction(x)) for x in self.steiner_points_x]
        self.steiner_points_y = [str(Fraction(y)) for y in self.steiner_points_y]

        print(f"Final Steiner Points: {self.steiner_points}")
        print(f"Final Edges: {edges}")

        return Cgshop2025Solution(
            content_type="CG_SHOP_2025_Solution",
            instance_uid=instance.instance_uid,
            steiner_points_x=self.steiner_points_x,
            steiner_points_y=self.steiner_points_y,
            edges=edges,
        )

import math
from itertools import combinations

from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance
from cgshop2025_pyutils.data_schemas.solution import Cgshop2025Solution
from cgshop2025_pyutils.geometry import Point, ConstrainedTriangulation
from scipy.spatial import Delaunay
import numpy as np
from cgshop2025_pyutils import (
    DelaunayBasedSolver,
    InstanceDatabase,
    ZipSolutionIterator,
    ZipWriter,
    verify,
)
import zipfile
import json

from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance


class OurSolution:
    def __init__(self, instance: Cgshop2025Instance):
        self.instance = instance
        self.steiner_points = []  # יצירת רשימה ריקה לנקודות שטיינר
        self.steiner_points_x = []
        self.steiner_points_y = []
        self.edges_added = []  # רשימה לצלעות חדשות שנוספו

    def calculate_angle(self, p1, p2, p3 ):
        def distance(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        # חישוב המרחקים בין הנקודות
        ab = distance(p1, p2)
        bc = distance(p2, p3)
        ca = distance(p3, p1)

        # בדיקת תקינות כדי למנוע חילוק באפס
        epsilon = 1e-9  # סף למרחק קטן מאוד
        if ab < epsilon or ca < epsilon or bc < epsilon:
            print(f"Warning: Points are too close or identical - p1={p1}, p2={p2}, p3={p3}")
            return 0.0  # אפשר להחזיר זווית ברירת מחדל או לדלג על המקרה

        # חישוב קוסינוס הזווית בעזרת נוסחת הקוסינוס
        cos_theta = (ab ** 2 + ca ** 2 - bc ** 2) / (2 * ab * ca)
        cos_theta = max(-1, min(1, cos_theta))  # למניעת בעיות נומריות מחוץ לטווח [-1, 1]

        # חישוב הזווית ברדיאנים והמרה למעלות
        angle = math.acos(cos_theta) * (180 / math.pi)
        return angle

    def find_perpendicular_intersection(self, vertex, p1, p2):
        """
        מחשבת את נקודת החיתוך של האנך היורד מהקודקוד vertex לצלע p1-p2 בצורה מדויקת.
        """
        x0, y0 = vertex
        x1, y1 = p1
        x2, y2 = p2

        # אם הצלע אנכית (x1 == x2)
        if abs(x2 - x1) < 1e-9:
            return x1, y0  # נקודה עם אותו X של הצלע ואותו Y של הקודקוד

        # אם הצלע אופקית (y1 == y2)
        if abs(y2 - y1) < 1e-9:
            return x0, y1  # נקודה עם אותו Y של הצלע ואותו X של הקודקוד

        # חישוב השיפוע של הצלע p1-p2
        m = (y2 - y1) / (x2 - x1)

        # השיפוע של האנך (ההופכי והנגדי)
        m_perpendicular = -1 / m

        # חישוב נקודת ההשקה (פתרון משוואות של שני ישרים)
        b1 = y1 - m * x1  # חיתוך צלע עם ציר Y
        b2 = y0 - m_perpendicular * x0  # חיתוך האנך עם ציר Y

        # מציאת נקודת החיתוך
        x_intersect = (b2 - b1) / (m - m_perpendicular)
        y_intersect = m * x_intersect + b1

        return x_intersect, y_intersect

    def ourFirstSol(self) -> Cgshop2025Solution:
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
        MAX_STEINER_POINTS = 100
        self.steiner_points_x = []
        self.steiner_points_y = []
        self.edges_added = []  # רשימה לצלעות החדשות שנוספו

        while triangles:
            t = triangles.pop()
            p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]

            angles = [
                (self.calculate_angle(p1, p2, p3), p1, p2, p3),
                (self.calculate_angle(p2, p3, p1), p2, p3, p1),
                (self.calculate_angle(p3, p1, p2), p3, p1, p2),
            ]
            angles.sort(reverse=True)

            if angles[0][0] > 90:
                print(f"Found obtuse triangle: {t}")
                vertex, p1, p2 = angles[0][1], angles[0][2], angles[0][3]
                if len(self.steiner_points_x) >= MAX_STEINER_POINTS:
                    print("Reached the maximum number of Steiner points. Stopping.")
                    break

                steiner_point = self.find_bisector_intersection(vertex, p1, p2)
                self.steiner_points.append(steiner_point)
                self.steiner_points_x.append(int(steiner_point[0]))
                self.steiner_points_y.append(int(steiner_point[1]))

                print(f"Added Steiner Point: {steiner_point}")

                # יצירת שני משולשים חדשים
                new_tri1 = (points.index(vertex), points.index(p1), len(points))
                new_tri2 = (points.index(vertex), len(points), points.index(p2))
                points.append(steiner_point)

                # שמירת הצלעות החדשות
                self.edges_added.append((points.index(vertex), len(points) - 1))
                self.edges_added.append((len(points) - 1, points.index(p1)))
                self.edges_added.append((len(points) - 1, points.index(p2)))

                triangles.add(new_tri1)
                triangles.add(new_tri2)

        print("Final Steiner Points:", self.steiner_points)
        print("Edges Added:", self.edges_added)

        print("Final Steiner Points:", self.steiner_points)
        print("Triangles:")
        for t in triangles:
            print(f"Triangle: {t} -> Points: {points[t[0]]}, {points[t[1]]}, {points[t[2]]}")
        return Cgshop2025Solution(
            instance_uid=instance.instance_uid,
            steiner_points_x=self.steiner_points_x,
            steiner_points_y=self.steiner_points_y,
            edges=edges + self.edges_added,
        )




import numpy as np

from scipy.special import comb

from shapely.wkt import loads
from shapely.geometry import LineString


class LineStringBender:

    __sample_number_value = 1000  # can be used like resolution curve

    def __init__(self, geometry, relative_distance_along_line, offset_distance_from_line, offset_position):
        """
        :param geometry: LineString shapely geometry
        :type geometry: shapely.geometry.LineString
        :param relative_distance_along_line: relative distance to put the node controller on the line ]0-1]
        :type relative_distance_along_line: float, int
        :param offset_distance_from_line: the offset distance from the line, like a buffer value (meters)
        :type offset_distance_from_line: float, int
        :param offset_distance_from_line: offset position, on the right/left of the line
        :type offset_distance_from_line: str, one of 'right', 'left'
        """

        assert geometry.geom_type == "LineString", f"Only Linestring are supported. {geometry.geom_type} found"
        assert len(geometry.coords[:]) == 2, f"Only a linestring containing 2 coordinates: {len(geometry.coords[:])} found"
        assert 0 < relative_distance_along_line <= 1, "Define the relative distance ]0 to 1]"
        assert isinstance(offset_distance_from_line, (float, int)), "Offset value should be a number"
        assert offset_position in {"right", "left"}, "Offset position: 'right' or 'left'"

        self._geom = geometry
        self._relative_distance_along_line = relative_distance_along_line
        self._offset_distance_from_line = offset_distance_from_line
        self._offset_position = offset_position

        self._prepare_input()
        self.__compute_smooth_curve()

    def smooth_curve_geom(self):
        return LineString(list(zip(self._x_vals, self._y_vals))[::-1])

    def raw_curve_geom(self):
        return LineString(self._geom_coordinates)

    def node_controler_geom(self):
        return self._point_control_on_line

    def _prepare_input(self):

        self._point_control_on_line = self._geom.interpolate(
            self._relative_distance_along_line,
            normalized=True
        )
        point_control_line = LineString([
            self._geom.coords[0],
            *self._point_control_on_line.coords[:]
        ])
        offset_line = point_control_line.parallel_offset(
            self._offset_distance_from_line,
            self._offset_position
        )
        if self._offset_position == "left":
            node_control = offset_line.coords[-1]
        elif self._offset_position == "right":
            node_control = offset_line.coords[0]
        else:
            raise ValueError("Should never happened")

        # insert the coord in the center of the line and create a numpy array
        geom_coordinates = self._geom.coords[:]
        geom_coordinates.insert(1, node_control)
        self._geom_coordinates = geom_coordinates
        self._lines_nodes = np.array(geom_coordinates)

    def __compute_smooth_curve(self):
        self._x_vals, self._y_vals = self.__bezier_curve(self._lines_nodes, sample_number=self.__sample_number_value)

    @staticmethod
    def __bernstein_poly(i, n, t):
        """
            source: https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
            The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def __bezier_curve(self, points, sample_number):
        """
           source: https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
           Given a set of control points, return the
           bezier curve defined by the control points.

           points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000

            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        n_points = len(points)
        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, sample_number)

        polynomial_array = np.array([
            self.__bernstein_poly(i, n_points - 1, t) for i in range(0, n_points)
        ])

        x_vals = np.dot(x_points, polynomial_array)
        y_vals = np.dot(y_points, polynomial_array)

        return x_vals, y_vals


if __name__ == "__main__":

    input_line = loads("LINESTRING(0 0, 25 25)")
    curve_process = LineStringBender(input_line, 0.5, 2, 'right')
    curve = curve_process.smooth_curve_geom()
    print(curve.wkt)
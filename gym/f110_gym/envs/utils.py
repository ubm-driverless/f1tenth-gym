import math
import numpy as np

class AngleOp:
    """
    Class to perform operations with angles. The angles are in radians and in the range [-pi, pi). All the operations
    are performed in radians.
    """

    def __init__(self):
        raise NotImplementedError('This class cannot be instantiated')

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize an angle to the range [-pi, pi).

        :param float angle: Angle to normalize in radians.
        :return: normalized angle.
        :rtype: float
        """
        # print(angle)
        angle = math.fmod(angle + math.pi, 2.0 * math.pi)
        if angle < 0:
            angle += 2.0 * math.pi
        return angle - math.pi

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """
        Computes `b - a` in the range [-pi, pi).

        :param float a: Angle [rad].
        :param float b: Angle [rad].
        :return: angle difference `b - a` [rad].
        :rtype: float
        """

        diff = math.fmod(b - a + math.pi, 2.0 * math.pi)
        if diff < 0:
            diff += 2.0 * math.pi
        return diff - math.pi

    @staticmethod
    def bisect_angle(a: float, b: float) -> float:
        """
        Finds the angle bisector between two angles. The inputs can be given in any order and they don't need to be
        normalized. The output is normalized.

        :param float a: First angle [rad].
        :param float b: Second angle [rad].
        :return: angle bisector between `a` and `b` [rad].
        :rtype: float
        """

        return AngleOp.normalize_angle(a + AngleOp.angle_diff(a, b) * 0.5)

    @staticmethod
    def weighted_circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
        """
        Computes the weighted circular mean of a set of angles, given their respective weights. The inputs don't need to
        be normalized. The output is in the range [-pi, pi).

        :param np.ndarray angles: Array of angles [rad].
        :param np.ndarray weights: Array of weights. ``weight[i]`` is associated to ``angles[i]``.
        :return: the weighted circular mean [rad].
        :rtype: float
        """

        if len(angles) != len(weights):
            raise ValueError('The number of angles and weights must be the same.')

        sum_sin = np.sum(np.sin(angles) * weights)
        sum_cos = np.sum(np.cos(angles) * weights)

        return AngleOp.normalize_angle(math.atan2(sum_sin, sum_cos))

    @staticmethod
    def cos_angle_diff(a: float, b: float) -> float:
        """
        Computes the cosine of the angle difference between two angles. The inputs don't need to be normalized.
        The order of the angles doesn't matter, since `cos(a - b) = cos(b - a)`.

        :param float a: Angle [rad].
        :param float b: Angle [rad].
        :return: returns the result of `cos(a - b)` [rad].
        :rtype: float
        """

        return math.cos(AngleOp.angle_diff(a, b))

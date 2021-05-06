# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x_min, y_min, x_max, y_max)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlbr, confidence, feature):
        self.tlbr = np.asarray(tlbr, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        return self.tlbr.copy()

    @property
    def tlwh(self):
        '''
        Convert to (x, y, w, h)
        '''
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlbr.copy()
        width = ret[2] - ret[0]
        height = ret[3] - ret[1]
        ret[:2] = (ret[:2] + ret[2:]) / 2
        ret[2] = width / height
        ret[3] = height
        return ret

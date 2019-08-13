#
#   Lightnet related data processing
#   Utilitary classes and functions for the data subpackage
#   Copyright EAVISE
#

from abc import ABC, abstractmethod
import shapely.geometry as shgeo

__all__ = ['Compose']


class Compose(list):
    """ This is lightnet's own version of :class:`torchvision.transforms.Compose`.

    Note:
        The reason we have our own version is because this one offers more freedom to the user.
        For all intends and purposes this class is just a list.
        This `Compose` version allows the user to access elements through index, append items, extend it with another list, etc.
        When calling instances of this class, it behaves just like :class:`torchvision.transforms.Compose`.

    Note:
        I proposed to change :class:`torchvision.transforms.Compose` to something similar to this version,
        which would render this class useless. In the meanwhile, we use our own version
        and you can track `the issue`_ to see if and when this comes to torchvision.

    Example:
        >>> tf = ln.data.transform.Compose([lambda n: n+1])
        >>> tf(10)  # 10+1
        11
        >>> tf.append(lambda n: n*2)
        >>> tf(10)  # (10+1)*2
        22
        >>> tf.insert(0, lambda n: n//2)
        >>> tf(10)  # ((10//2)+1)*2
        12
        >>> del tf[2]
        >>> tf(10)  # (10//2)+1
        6

    .. _the issue: https://github.com/pytorch/vision/issues/456
    """
    def __call__(self, data, boxes, classes):
        for tf in self:
            data, boxes, classes = tf(data, boxes, classes)
        return data, boxes, classes

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ['
        for tf in self:
            format_string += '\n  {tf}'
        format_string += '\n]'
        return format_string


class BaseTransform(ABC):
    """ Base transform class for the pre- and post-processing functions.
    This class allows to create an object with some case specific settings, and then call it with the data to perform the transformation.
    It also allows to call the static method ``apply`` with the data and settings. This is usefull if you want to transform a single data object.
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, data, boxes, classes):
        return self.apply(data, **self.__dict__), boxes, classes

    @classmethod
    @abstractmethod
    def apply(cls, data, **kwargs):
        """ Classmethod that applies the transformation once.

        Args:
            data: Data to transform (eg. image)
            **kwargs: Same arguments that are passed to the ``__init__`` function
        """
        return data


class BaseMultiTransform(ABC):
    """ Base multiple transform class that is mainly used in pre-processing functions.
    This class exists for transforms that affect both images and annotations.
    It provides a classmethod ``apply``, that will perform the transormation on one (data, target) pair.
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @abstractmethod
    def __call__(self, data):
        return data

    @classmethod
    def apply(cls, data, target=None, **kwargs):
        """ Classmethod that applies the transformation once.

        Args:
            data: Data to transform (eg. image)
            target (optional): ground truth for that data; Default **None**
            **kwargs: Same arguments that are passed to the ``__init__`` function
        """
        obj = cls(**kwargs)
        res_data = obj(data)

        if target is None:
            return res_data

        res_target = obj(target)
        return res_data, res_target
    
    def set_value(self, key, value):
        setattr(self, key, value) 


# modify for rotate object
def choose_best_pointorder_fit_another(wait_fit_poly1, ori_poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = wait_fit_poly1[0]
    y1 = wait_fit_poly1[1]
    x2 = wait_fit_poly1[2]
    y2 = wait_fit_poly1[3]
    x3 = wait_fit_poly1[4]
    y3 = wait_fit_poly1[5]
    x4 = wait_fit_poly1[6]
    y4 = wait_fit_poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(ori_poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def calchalf_iou(poly1, poly2):
    """
        poly1: label
        poly2: imgbox
        It is not the iou on usual, the iou is the value of intersection over poly1
    """
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    poly1_area = poly1.area
    half_iou = inter_area / poly1_area
    return inter_poly, half_iou

# def saveimagepatches(img, subimgname, left, up):
#     subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
#     outdir = os.path.join(self.outimagepath, subimgname + self.ext)
#     cv2.imwrite(outdir, subimg)

def GetPoly4FromPoly5(poly):
    distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
    distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
    pos = np.array(distances).argsort()[0]
    count = 0
    outpoly = []
    while count < 5:
        #print('count:', count)
        if (count == pos):
            outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
            outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
            count = count + 1
        elif (count == (pos + 1)%5):
            count = count + 1
            continue

        else:
            outpoly.append(poly[count * 2])
            outpoly.append(poly[count * 2 + 1])
            count = count + 1
    return outpoly
 

def polyorig2Crop(left, up, poly, right=0, down=0):
    polyInsub = np.zeros(len(poly))
    for i in range(int(len(poly)/2)):
        polyInsub[i * 2] = int(poly[i * 2] - left)
        if (polyInsub[i * 2] <= 1):
            polyInsub[i * 2] = 1
        elif (polyInsub[i * 2] >= right):
            polyInsub[i * 2] = right

        polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        if (polyInsub[i * 2 + 1] <= 1):
            polyInsub[i * 2 + 1] = 1
        elif (polyInsub[i * 2 + 1] >= down):
            polyInsub[i * 2 + 1] = down

    return polyInsub

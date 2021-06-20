from PIL import Image
import numpy as np

vocab = {
    'width': {
        'null': [-1],
        'very small': [0],
        'small': [1],
        'medium': [2],
        'large': [3],
        'very large': [4],
        'huge': [5],
    },
    'height': {
        'null': [-1],
        'very small': [0],
        'small': [1],
        'medium': [2],
        'large': [3],
        'very large': [4],
        'huge': [5],
    },
    'fill': {
        'null': [-1, -1, -1, -1],
        'yes': [1, 1, 1, 1],
        'no': [0, 0, 0, 0],
        'right-half': [0, 1, 0, 1],
        'left-half': [1, 0, 1, 0],
        'top-half': [1, 1, 0 ,0],
        'bottom-half': [0, 0, 1, 1],
    },
    'shape': {
        'null': [-1, -1],
        'triangle': [0, 3],
        'right triangle': [1, 3],
        'square': [2, 4],
        'rectangle': [3, 4],
        'diamond': [4, 4],
        'pentagon': [5, 5],
        'octagon': [6, 8],
        'star': [7, 10],
        'plus': [8, 12],
        'circle': [9, 0],
        'pac-man': [10, 0],
        'heart': [11, 0],
    },
    'angle': [],
    'inside': [1],
    'above': [1],
    'left-of': [1],
    'alignment': {
        'null': [-1, -1, -1, -1],
        'top-left': [1, 0, 0, 0],
        'top-right': [0, 1, 0, 0],
        'bottom-left': [0, 1, 1, 0],
        'bottom-right': [0, 0, 0, 1],
    }
}

weights = np.asarray([
        0.500, # width
        0.500, # height
        0.200, # fill1
        0.200, # fill2
        0.200, # fill3
        0.200, # fill4
        1.000, # shapeID
        0.500, # shapeSides
        0.750, # angle
        0.200, # inside
        0.200, # above
        0.200, # left-of
        0.125, # align1
        0.125, # align2
        0.125, # align3
        0.125, # align4
    ])



def attrGen(nodeAttrs, attrName, attrVal):
    if attrName == 'shape':
        nodeAttrs[attrName] = Shape.convert(attrVal)
    elif attrName == 'width':
        nodeAttrs[attrName] = Width.convert(attrVal)
    elif attrName == 'height':
        nodeAttrs[attrName] = Height.convert(attrVal)
    elif attrName == 'angle':
        nodeAttrs[attrName] = Angle.convert(attrVal)
    elif attrName == 'alignment':
        nodeAttrs[attrName] = Alignment.convert(attrVal)
    elif attrName == 'fill':
        nodeAttrs[attrName] = Fill.convert(attrVal)
    else:
        raise ValueError('Unidentifiable Attribute found', attrName, attrVal)


class Shape:

    @staticmethod
    def convert(shapeName):
        return Shape(1, {'name': shapeName})

    def __init__(self, status, props):
        self.status = status
        self.props = props

    def __sub__(self, other):
        if other == 0:
            return Shape(self.status, self.props.copy())

        newStatus = self.status - other.status
        if newStatus:
            return Shape(newStatus, self.props.copy())
        elif self.props == other.props:
            return 0
        else:
            return Shape(newStatus, {'before': other.props['name'], 'after': self.props['name']})

    def __rsub__(self, other):
        if other == 0:
            return Shape(-1, self.props.copy())
        else:
            raise ValueError('Shape Subtraction: Inconsistent Types!', self, other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{0}, {1}'.format(self.status, self.props)

    def __repr__(self):
        return self.__str__()


# Width Attribute Wrapper
class Width:
    scale2Num = {'very small': 0, 'small': 1, 'medium': 2, 'large': 3, 'very large': 4,'huge': 5}
    # num2Scale = ['very small', 'small', 'medium', 'large', 'very large', 'huge']

    @staticmethod
    def convert(widthName):
        return Width(1, {'width': Width.scale2Num[widthName]})

    def __init__(self, status, props):
        self.status = status
        self.props = props

    def __sub__(self, other):
        if other == 0:
            return Width(self.status, self.props.copy())

        newStatus = self.status - other.status
        if newStatus:
            return Width(newStatus, self.props.copy())
        elif self.props == other.props:
            return 0
        else:
            return Width(0, {'width': self.props['width'] - other.props['width']})


    def __rsub__(self, other):
        if other == 0:
            return Width(-1, self.props.copy())
        raise ValueError('Width Subtraction: Inconsistent Types!', self, other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{0}, {1}'.format(self.status, self.props)

    def __repr__(self):
        return self.__str__()




# Height Attribute Wrapper
class Height:
    scale2Num = {'very small': 0, 'small': 1, 'medium': 2, 'large': 3, 'very large': 4,'huge': 5}
    # num2Scale = ['very small', 'small', 'medium', 'large', 'very large', 'huge']

    @staticmethod
    def convert(heightName):
        return Height(1, {'height': Height.scale2Num[heightName]})

    def __init__(self, status, props):
        self.status = status
        self.props = props

    def __sub__(self, other):
        if other == 0:
            return Height(self.status, self.props.copy())

        newStatus = self.status - other.status
        if newStatus:
            return Height(newStatus, self.props.copy())
        elif self.props == other.props:
            return 0
        else:
            return Height(0, {'height': self.props['height'] - other.props['height']})


    def __rsub__(self, other):
        if other == 0:
            return Height(-1, self.props.copy())
        raise ValueError('Height Subtraction: Inconsistent Types!', self, other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{0}, {1}'.format(self.status, self.props)

    def __repr__(self):
        return self.__str__()

# Angle Attribute
class Angle:

    @staticmethod
    def convert(angle):
        return Angle(1, {'angle': int(angle)})

    def __init__(self, status, props):
        self.status = status
        self.props = props

    def __sub__(self, other):
        if other == 0:
            return Angle(self.status, self.props.copy())

        newStatus = self.status - other.status

        if newStatus:
            return Angle(newStatus, self.props.copy())
        elif self.props == other.props:
            return 0
        else:
            return Angle(0, {'angle': (self.props['angle'] - other.props['angle'])})


    def __rsub__(self, other):
        if other == 0:
            return Angle(-1, self.props.copy())
        raise ValueError('Angle Subtraction: Inconsistent Types!', self, other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{0}, {1}'.format(self.status, self.props)

    def __repr__(self):
        return self.__str__()


class Alignment: # top, bottom
    align2Num = {'top': 1, 'bottom': 0, 'right': 1, 'left': 0}

    @staticmethod
    def convert(alignName):
        alignAxes = alignName.split('-')
        vert = Alignment.align2Num[alignAxes[0]]
        horiz = Alignment.align2Num[alignAxes[1]]
        return Alignment(1, {'vertical': vert, 'horizontal': horiz})

    def __init__(self, status, props):
        self.status = status
        self.props = props

    def __sub__(self, other):
        if other == 0:
            return Alignment(self.status, self.props.copy())

        newStatus = self.status - other.status

        if newStatus:
            return Alignment(newStatus, self.props.copy())
        elif self.props == other.props:
            return 0;
        else:
            return Alignment(0, {'vertical': self.props['vertical'] - other.props['vertical'], 'horizontal': self.props['horizontal'] - other.props['horizontal']})

    def __rsub__(self, other):
        if other == 0:
            return Alignment(-1, self.props.copy())
        raise ValueError('Alignment Subtraction: Inconsistent Types!', self, other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{0}, {1}'.format(self.status, self.props)

    def __repr__(self):
        return self.__str__()

class Fill:
    # 1 = quadrant filled, 0 = quadrant empty, quadrants represented left to right
    fill2Num = {'yes': (1, 1, 1, 1), 'no': (0, 0, 0, 0), 'right-half': (0, 1, 0, 1), 'left-half': (1, 0, 1, 0), 'top-half': (1, 1, 0 ,0), 'bottom-half': (0, 0, 1, 1)}

    @staticmethod
    def convert(fillYN):
        return Fill(1, {'fill': Fill.fill2Num[fillYN]})

    def __init__(self, status, props):
        self.status = status
        self.props = props

    def __sub__(self, other):
        if other == 0:
            return Fill(self.status, self.props.copy())

        newStatus = self.status - other.status

        if newStatus:
            return Fill(newStatus, self.props.copy())
        elif self.props == other.props:
            return 0
        else:
            newFill = tuple(np.subtract(self.props['fill'], other.props['fill']))
            return Fill(0, {'fill': newFill})

    def __rsub__(self, other):
        if other == 0:
            return Fill(-1, self.props.copy())
        raise ValueError('Fill Subtraction: Inconsistent Types!', self, other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{0}, {1}'.format(self.status, self.props)

    def __repr__(self):
        return self.__str__()
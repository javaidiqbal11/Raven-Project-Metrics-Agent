import numpy as np
from PIL import Image

# set of transformation functions that return the confidence of said transformation


# check to see if whole image is the same
identical = lambda imgA, imgB, threshold: np.count_nonzero(imgA - imgB) < threshold

# check to see if whole image is flipped vertically
vertFlip = lambda imgA, imgB, threshold: np.count_nonzero(imgA - np.flipud(imgB)) < threshold

# check to see if whole image is flipped horizontally
horizFlip = lambda imgA, imgB, threshold: np.count_nonzero(imgA - np.fliplr(imgB)) < threshold

# check to see if whole image is rotated 90deg
rot90 = lambda imgA, imgB, threshold: np.count_nonzero(imgA - np.rot90(imgB, k=1)) < threshold

# check to see if whole image is rotated 180deg
rot180 = lambda imgA, imgB, threshold: np.count_nonzero(imgA - np.rot90(imgB, k=2)) < threshold

# check to see if whole image is rotated 270deg
rot270 = lambda imgA, imgB, threshold: np.count_nonzero(imgA - np.rot90(imgB, k=3)) < threshold

# export 2x2 rule functions
ruleFuncs = {'identical': identical, 'verFlip': vertFlip, 'horizFlip': horizFlip, 'rot90': rot90, 'rot180': rot180,
             'rot270': rot270}

# RULE FUNCTIONS (use graphical location/context) identical3 = lambda imgA, imgB, imgC, threshold:
# np.count_nonzero(imgA - imgB) < threshold and np.count_nonzero(imgA - imgC) < threshold and np.count_nonzero(imgC -
# imgB) < threshold

# quantity based addition
addQuantity3 = lambda imgA, imgB, imgC, imgSize: np.count_nonzero(imgA == 0) + np.count_nonzero(
    imgB == 0) - np.count_nonzero(imgC == 0) < imgSize * 0.08


# location based addition VERIFIED
def or3(imgA, imgB, imgC, imgSize):
    added = 255 - ((255 - imgA) + (255 - imgB))
    added[added > 255] = 255
    added[added < 0] = 0

    return np.count_nonzero(added != imgC) < (imgSize * 0.08)


# location based XOR VERIFIED
def xor3(imgA, imgB, imgC, imgSize):
    diffs = 255 - ((imgA != imgB) * 255)
    return np.count_nonzero(diffs != imgC) < (imgSize * 0.08)


# and3 VERIFIED
def and3(imgA, imgB, imgC, imgSize):
    # imgA = 255 - imgA
    # imgB = 255 - imgB
    # mult = imgA * imgB
    # mult[mult > 255] = 255
    # mult = 255 - mult
    anded = 255 - (((imgA == 0) & (imgB == 0)) * 255)
    return np.count_nonzero(anded - imgC) < (imgSize * 0.08)


ruleFuncs3 = {'or3': or3, 'xor3': xor3, 'and3': and3, 'addQuantity3': addQuantity3}
# def xor3(imgA, imgB, imgC, imgSize):
#     zA = imgA / 255
#     zB = imgB / 255
#     zC = imgC / 255

#     return 0


# Image.fromarray(np.uint8(x)).show()
# from PIL import Image
# Image.fromarray(np.uint8(mask)).show()

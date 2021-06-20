# DO NOT MODIFY THIS FILE.
#
# Any modifications to this file will not be used when grading your project.
# If you have any questions, please email the TAs.

# A single figure in a Raven's Progressive Matrix problem, comprised of a name
# and a list of RavensObjects.

from PIL import Image, ImageDraw
import copy
from RavensObject import Object


class Figure:

    # Creates a new figure for a Raven's Progressive Matrix given a name.
    #
    # Your agent does not need to use this method.
    #
    # @param name the name of the figure

    def __init__(self, image_source):
        # The name of the figure. The name of the figure will always match
        # the HashMap key for this figure.
        #
        # The figures in the problem will be named A, B, and C for 2x1 and 2x2
        # problems. The figures in the problem will be named A, B, C, D, E, F, G,
        # and H in 3x3 problems. The first row is A, B, and C; the second row is
        # D, E, and F; and the third row is G and H.
        #
        # Answer options will always be named 1 through 6.
        #
        # The numbers for the answer options will be randomly generated on each run
        # of the problem. The correct answer will remain the same, but its number
        # will change.
        if isinstance(image_source, str):
            # Load image from file and make it black or white (no grey!)
            self.image = Image.open(image_source).convert('L').point(lambda x: 0 if x < 255 else 255, '1')
            # self.image = Image.open(image_source).convert('L').point(lambda x: 0 if x == 0 else 255, '1')
        elif isinstance(image_source, Image.Image):
            self.image = image_source.convert('L').point(lambda x: 0 if x < 255 else 255, '1')

        self.pixels = self.convert_image_to_pixels_array(self.image)
        self.objects = []

    def convert_image_to_pixels_array(self, image):
        pixels = list(image.getdata())
        width, height = image.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

        return pixels

    # Identifies all contiguous figures in these images
    # Only includes black objects from the image
    # ((Assumes images are comprised of black and white pixels ONLY. no grey!))
    def identify_objects(self):
        im = copy.deepcopy(self.image)
        width, height = im.size

        dark_fill_val = 1
        light_fill_val = 254
        for x in range(width):
            for y in range(height):
                xy = (x, y)
                l_val = im.getpixel(xy)

                if l_val == 0:
                    ImageDraw.floodfill(im, xy, dark_fill_val)
                    self.objects.append(Object(xy, dark_fill_val))
                    dark_fill_val += 1
                elif l_val == 255:
                    ImageDraw.floodfill(im, xy, light_fill_val)
                    light_fill_val -= 1
                else:
                    for obj in self.objects:
                        if obj.l_val == l_val:
                            obj.add_pixel(xy)
                            break

    def find_centroids(self):
        for obj in self.objects:
            obj.find_centroid()

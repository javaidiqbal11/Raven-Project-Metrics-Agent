# DO NOT MODIFY THIS FILE.
#
# Any modifications to this file will not be used when grading your project.
# If you have any questions, please email the TAs.

# A single object in a RavensFigure -- typically, a single shape in a frame,
# such as a triangle or a circle -- comprised of a list of name-value attributes.

from PIL import Image, ImageFilter, ImageChops, ImageStat


class Object:
    # Constructs a new RavensObject given a name.
    #
    # Your agent does not need to use this method.
    #
    # @param name the name of the object

    threshold = .96
    centroid = None
    max_x = 0
    min_x = 9999
    max_y = 0
    min_y = 9999
    is_filled = None
    image = None

    def __init__(self, xy, l_val):
        # The name of this RavensObject. Names are assigned starting with the
        # letter Z and proceeding backwards in the alphabet through the objects
        # in the Frame. Names do not imply correspondence between shapes in
        # different frames. Names are simply provided to allow agents to organize
        # their reasoning over different figures.
        #
        # Within a RavensFigure, each RavensObject has a unique name.

        self.l_val = l_val
        self.area = [xy]

        # A dictionary of name-value pairs representing this object's attributes. The key
        # for these attributes is the name of the attribute, and the value is the
        # value of the attribute. For example, a filled large square would have
        # attribute pairs "shape:square", "size:large", and "filled:yes".

    def __eq__(self, other):
        if self.image is None:
            self.make_image()
        if other.image is None:
            other.make_image()

        if self.is_filled != other.is_filled:
            return False

        max_similarity = 0
        for x_offset in range(-3, 4, 1):
            for y_offset in range(-3, 4, 1):
                diff = ImageChops.difference(ImageChops.offset(self.image, x_offset, y_offset), other.image)
                num_pixels = max(self.image.size[0], other.image.size[0]) * max(self.image.size[1], other.image.size[1])
                diff_stats = ImageStat.Stat(diff)
                similarity = 1.0 - ((diff_stats.sum[0] / 255) / num_pixels)
                max_similarity = max(similarity, max_similarity)
        return max_similarity >= self.threshold

    def __ne__(self, other):
        return not self == other

    def add_pixel(self, xy):
        self.area.append(xy)
        if xy[0] > self.max_x:
            self.max_x = xy[0]
        if xy[0] < self.min_x:
            self.min_x = xy[0]
        if xy[1] > self.max_y:
            self.max_y = xy[1]
        if xy[1] < self.min_y:
            self.min_y = xy[1]

    def remove_pixel(self, xy):
        self.area.remove(xy)

    def find_centroid(self):
        x_total = 0
        y_total = 0

        for pixel in self.area:
            x_total += pixel[0]
            y_total += pixel[1]

        x_cen = round(x_total / len(self.area), 0)
        y_cen = round(y_total / len(self.area), 0)

        self.centroid = (x_cen, y_cen)
        return self.centroid

    def size(self):
        width = self.max_x - self.min_x + 1
        height = self.max_y - self.min_y + 1
        return width, height

    def make_image(self):
        self.image = Image.new('L', self.size(), color=255)
        for xy in self.area:
            try:
                translated_xy = xy[0] - self.min_x, xy[1] - self.min_y
                self.image.putpixel(translated_xy, 0)
            except IndexError:
                pass

        # If centroid is located in the filled in area, we know the object is filled
        if self.centroid is None:
            self.find_centroid()
        if self.find_centroid() in self.area:
            self.is_filled = True
        else:
            self.is_filled = False

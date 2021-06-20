# I import the Figure file where set the figure properties
from RavensFigure import Figure
from PIL import Image

# I import the object file
from RavensObject import Object
import copy


def figure_and(fig1, fig2):
    im1 = fig1.image
    im2 = fig2.image
    if im1.size != im2.size:
        raise Exception('All the images should be equal size for AND operation')
    size = im1.size
    image = Image.new('L', size, color=255)

    for x in range(size[0]):
        for y in range(size[1]):
            xy = x, y
            if im1.getpixel(xy) == im2.getpixel(xy) == 0:
                image.putpixel(xy, 0)
    return Figure(image)


def figure_xor(fig1, fig2):
    im1 = fig1.image
    im2 = fig2.image
    if im1.size != im2.size:
        raise Exception('All the images should be equal size for XOR operation')
    size = im1.size
    image = Image.new('L', size, color=255)

    for x in range(size[0]):
        for y in range(size[1]):
            xy = x, y
            if im1.getpixel(xy) != im2.getpixel(xy):
                image.putpixel(xy, 0)
    return Figure(image)


def figure_add(fig1, fig2):
    if fig1.image.size != fig2.image.size:
        raise Exception('Figures must be same size to SUBTRACT them')

    size = fig1.image.size
    image = Image.new('L', size, color=255)

    for obj1 in fig1.objects:
        for xy in obj1.area:
            image.putpixel(xy, 0)

    for obj2 in fig2.objects:
        for xy in obj2.area:
            image.putpixel(xy, 0)

    return Figure(image)


def figure_subtract(fig1, fig2):
    if fig1.image.size != fig2.image.size:
        raise Exception('Figures should be same size to SUBTRACT them')

    image = copy.deepcopy(fig1.image)

    for obj2 in fig2.objects:
        for xy in obj2.area:
            image.putpixel(xy, 255)

    return Figure(image)


# Here we can test our model by changing the problems images with their notations

fig_b1_a = Figure("Problems/Basic Problems B/Basic Problem B-01/A.png")
fig_b1_b = Figure("Problems/Basic Problems B/Basic Problem B-01/B.png")

print('Objects are being identifying.')
fig_b1_a.identify_objects()
fig_b1_b.identify_objects()

print('Computation using mathemathics terms:')
figure_and(fig_b1_a, fig_b1_b)
figure_xor(fig_b1_a, fig_b1_b)
figure_add(fig_b1_b, fig_b1_a)
figure_subtract(fig_b1_b, fig_b1_a)

print('This is doing best now: \n')

fig_a = Figure("Problems/Basic Problems B/Basic Problem B-01/A.png")
fig_b = Figure("Problems/Basic Problems B/Basic Problem B-01/B.png")
fig_c = Figure("Problems/Basic Problems B/Basic Problem B-01/C.png")


figs = [fig_a, fig_b, fig_c]

for fig in figs:
    fig.identify_objects()

if fig_a.objects[0] == fig_b.objects[0]:
    print('A and B are equal')

else:
    if fig_a.objects[0] == fig_c.objects[0]:
        print('A and C are equal')

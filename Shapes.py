import math
import numpy as np

class VisShape:
    deltas = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))

    def spanArea(self, arr, r, c):
        if np.any(arr[r,c] == 0):
            self.netR += r
            self.netC += c
            # self.centroid
            self.top = min(r, self.top)
            self.bottom = max(r, self.bottom)
            self.left = min(c, self.left)
            self.right = max(c, self.right)
            arr[r,c] = 255
            return 1 + sum([self.spanArea(arr, r + dr, c + dc) for dr, dc in ((-1,-1), (-1, 0), (-1, 1), (1,-1), (1,0), (1,1))])
            # continue dfs
        else:
            return 0

    def dfs(self, arr, r, c):
        # print('launching dfs')
        visited, stack = set(), [(r,c)]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                # print('\texpanding', vertex)
                visited.add(vertex)
                r, c = vertex
                if np.any(arr[r,c] == 0):
                    self.area += 1
                    self.netR += r
                    self.netC += c
                    self.top = min(r, self.top)
                    self.bottom = max(r, self.bottom)
                    self.left = min(c, self.left)
                    self.right = max(c, self.right)
                    arr[r,c] = 255

                stack.extend([(r+dr, c+dc) for dr,dc in VisShape.deltas if (np.any(arr[r+dr,c+dc]==0) and (r+dr, c+dc) not in visited)])
                # stack.extend(graph[vertex] - visited)
        return visited



    def __init__(self, r, c, arr):
        self.top = r
        self.bottom = r
        self.left = c
        self.right = c

        self.netR = 0
        self.netC = 0
        self.area = 0

        self.points = self.dfs(arr, r, c)

        self.centroid = (self.netR / self.area, self.netC / self.area)
        boundingRectArea = ((self.bottom - self.top) * (self.right - self.left))
        # if boundingRectArea == 0:
            # print('zero area error')
            # import ipdb; ipdb.set_trace()
        self.rectangularity = self.area / boundingRectArea

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((self.rectangularity, self.area))

    def __eq__(self, other):
        return self.rectangularity == other.rectangularity and self.area == other.area

    def __ne__(self, other):
        return not(self == other)

    def __repr__(self):
        return '\trectangularity: {0}, area: {1}, top-left: ({2},{3})\n'.format(self.rectangularity, self.area * 4, self.top * 2, self.left * 2)

    # def rectangularity(self):
    #     return self.area / ((self.bottom - self.top) * (self.right - self.left))

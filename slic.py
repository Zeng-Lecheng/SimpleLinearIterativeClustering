from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from utils import rgb2lab_single, lab2rgb_single


class SlicElement:
    """
    Represents a single pixel in the image, or a centroid.
    """
    def __init__(self, data: np.ndarray):
        """
        :param data: vector of the pixel, in the form of [l, a, b, x, y] where x and y are positions in the image
        """
        assert data.shape == (5,)
        self.lab: np.ndarray = data
        self.seg = -1   # the segmentation this pixel belongs to, -1 for not segmented or centroids

    @property
    def color(self) -> np.ndarray:
        """
        Read lab color out
        :return: lab color
        """
        return self.lab[:3]

    @color.setter
    def color(self, value: np.ndarray):
        """
        Set color from outside
        :param value: lab color of the pixel, in the form of [l, a, b]
        :return:
        """
        assert value.shape == (3,)
        self.lab[:3] = value

    @property
    def pos(self) -> np.ndarray:
        """
        Read position out
        :return: position of the pixel, [x, y]
        """
        return self.lab[3:]


class Slic:
    def __init__(self, img: Image, k: int):
        """
        :param img: PIL.Image object, input image
        :param k: number of segments
        """
        self.img = np.array(img)
        self.k = k  # number of superpixels
        self.s = np.int(np.sqrt(self.img.shape[0] * self.img.shape[1] / k))     # see section 3.1
        self.m = 10     # see the ending paragraph of section 3.1
        self.shape = self.img.shape
        self.pixels = self._rgb2lab(self.img / 255.)  # ndarray of SlicElements, holds all pixels
        self.centroids: list[SlicElement] = []      # empty list waiting to hold centroids

    def fit(self, max_iter: int, converge_threshold: float):
        """
        Generate segments
        :param max_iter: maximum number of iterations
        :param converge_threshold: if distance of centroids moving from previous position is less than this value
                                   then skip following iterations
        :return: None
        """
        init_grid = np.mgrid[0: self.shape[0]: self.s, 0: self.shape[1]: self.s].T
        init_grid = init_grid.reshape(-1, 2).astype(int)

        # regular grid init
        for p in init_grid:
            self.centroids.append(self.pixels[p[0], p[1]])
        print('Initialization completed')

        # move to the lowest gradient position in a 3 × 3 neighborhood
        for m, c in enumerate(self.centroids):
            lowest_grad_ele: SlicElement = c
            lowest_grad = self.gradient(tuple(c.pos.astype(int)))

            # determine the 3 x 3 neighborhood for searching minimum
            lower_x, upper_x = int(max(c.pos[0] - 1, 0)), int(min(c.pos[0] + 1, self.shape[0] - 1))
            lower_y, upper_y = int(max(c.pos[1] - 1, 0)), int(min(c.pos[1] + 1, self.shape[1] - 1))
            for i in range(lower_x, upper_x + 1):
                for j in range(lower_y, upper_y + 1):
                    if (cur_grad := self.gradient((i, j))) < lowest_grad:   # if gradient at some point is lower
                        lowest_grad = cur_grad
                        lowest_grad_ele = self.pixels[i, j]
            self.centroids[m] = lowest_grad_ele
        print('Moving to the lower gradient in neighborhood completed')

        print('Updating centroids')
        # main iterations
        for i in tqdm(range(max_iter)):
            for p in np.nditer(self.pixels, flags=['refs_ok']):     # use nditer to reduce nested for loop
                # Assign the best matching pixels from a 2S × 2S square neighborhood
                pixel = p.item()
                nearest_centroid = self.centroids[0]    # initialization for finding the nearest centroid
                for j, c in enumerate(self.centroids):
                    # nearest and in 2S x 2S neighborhood
                    if self.distance(pixel, c) <= self.distance(pixel, nearest_centroid) and \
                            c.pos[0] - pixel.pos[0] < 2 * self.s and c.pos[1] - pixel.pos[1] < 2 * self.s:
                        nearest_centroid = c
                        pixel.seg = j   # update segment of the pixel

            new_centroids: list[SlicElement] = [SlicElement(np.zeros(5,)) for _ in range(len(self.centroids))]
            new_partition_cnt = Counter()  # count number in that partition for averaging
            # update centroids
            for p in np.nditer(self.pixels, flags=['refs_ok']):
                pixel = p.item()
                if pixel.seg != -1:
                    new_centroids[pixel.seg].lab += pixel.lab   # sum up pixels of each centroid
                    new_partition_cnt[pixel.seg] += 1
            for m, c in enumerate(new_centroids):
                if new_partition_cnt[m] > 0:
                    new_centroids[m].lab = new_centroids[m].lab / new_partition_cnt[m]      # averaging
                else:
                    new_centroids[m].lab = self.centroids[m].lab

            error = sum(
                [np.linalg.norm(self.centroids[k].lab - new_centroids[k].lab) for k in range(len(self.centroids))]
            ) / len(self.centroids)     # difference from previous positions
            self.centroids = new_centroids

            if error <= converge_threshold:
                break

        # enforce connections
        # move pixels into the segment which most of its neighbor are of
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                lower_x, upper_x = int(max(i - 1, 0)), int(min(i + 1, self.shape[0] - 1))
                lower_y, upper_y = int(max(j - 1, 0)), int(min(j + 1, self.shape[1] - 1))

                counter = Counter()
                for k in range(lower_x, upper_x + 1):
                    for m in range(lower_y, upper_y + 1):
                        counter[self.pixels[k, m].seg] += 1
                most_seg = counter.most_common(1)[0][0]
                self.pixels[i, j].seg = most_seg
        print('Enforce connections completed')

        print('Segmentation completed')

    def save(self, path: str):
        """
        Render each pixel with the color of its centroid, which is the average color of its segment.
        Then convert lab to rgb and save to path
        :param path: the path to save the image
        :return: None
        """
        rendered = np.zeros((self.shape[0], self.shape[1], 3))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                rendered[i, j] = self.centroids[self.pixels[i, j].seg].color

        rgb = self._lab2rgb(rendered) * 255.
        img = Image.fromarray(rgb.astype(np.uint8))
        img.save(path)
        print(f'Saved to {path}')


    def gradient(self, pos: tuple[int, int]) -> float:
        """
        Gradient at a given position. Eq. 2
        :param pos:
        :return:
        """
        lower_x, upper_x = max(pos[0] - 1, 0), min(pos[0] + 1, self.shape[0] - 1)
        lower_y, upper_y = max(pos[1] - 1, 0), min(pos[1] + 1, self.shape[1] - 1)
        return np.linalg.norm(
            self.pixels[upper_x, pos[1]].color - self.pixels[lower_x, pos[1]].color
        ) ** 2 + np.linalg.norm(
            self.pixels[pos[0], upper_y].color - self.pixels[pos[0], lower_y].color
        ) ** 2

    def distance(self, a: SlicElement, b: SlicElement) -> float:
        """
        Distance measure defined by Eq. 1
        :param a:
        :param b:
        :return:
        """
        return np.linalg.norm(a.color - b.color) + self.m / self.s * np.linalg.norm(a.pos - b.pos)

    @staticmethod
    def _rgb2lab(rgb: np.ndarray) -> np.ndarray:
        """
        Convert rgb images to lab color space
        :param rgb: rgb image
        :return: lab image
        """
        assert rgb.shape[2] == 3  # d.shape = (h, w, 3)

        lab = np.full(rgb.shape[:2], SlicElement(np.zeros(5, )))
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                lab_single = convert_color(sRGBColor(rgb[i, j][0], rgb[i, j][1], rgb[i, j][2]), LabColor, target_illuminant='d50')
                lab[i, j] = SlicElement(np.array([lab_single.lab_l, lab_single.lab_a, lab_single.lab_b, i, j]))

        return lab

    @staticmethod
    def _lab2rgb(lab: np.ndarray):
        """
        Convert lab images to rgb color space
        :param lab: lab image
        :return: rgb image in 0-1
        """
        assert lab.shape[2] == 3
        rgb = np.zeros(lab.shape)
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                rgb_ij = convert_color(LabColor(lab[i, j][0], lab[i, j][1], lab[i, j][2]), sRGBColor, target_illuminant='d50')
                rgb[i, j] = [rgb_ij.rgb_r, rgb_ij.rgb_g, rgb_ij.rgb_b]
        return rgb

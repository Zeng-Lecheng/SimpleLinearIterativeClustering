import numpy as np


def rgb2lab_single(rgb: np.ndarray):
    # convert a single pixel from rgb to lab
    # https://web.archive.org/web/20111111073625/http://www.easyrgb.com/index.php?X=MATH&H=02#text2
    assert rgb.shape == (3,)

    rgb_r, rgb_g, rgb_b = rgb
    rgb_r /= 255
    rgb_g /= 255
    rgb_b /= 255

    f = lambda x: ((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92
    rgb_r = f(rgb_r)
    rgb_g = f(rgb_g)
    rgb_b = f(rgb_b)

    rgb_r *= 100
    rgb_g *= 100
    rgb_b *= 100

    xyz_x = rgb_r * 0.4124 + rgb_g * 0.3576 + rgb_b * 0.1805
    xyz_y = rgb_r * 0.2126 + rgb_g * 0.7152 + rgb_b * 0.0722
    xyz_z = rgb_r * 0.0193 + rgb_g * 0.1192 + rgb_b * 0.9502

    xyz_x /= 95.047
    xyz_y /= 100
    xyz_z /= 108.883

    g = lambda x: x ** (1 / 3) if x > 0.008856 else (7.787 * x) + 16 / 116

    xyz_x = g(xyz_x)
    xyz_y = g(xyz_y)
    xyz_z = g(xyz_z)

    lab_l = 116 * xyz_y - 16
    lab_a = 500 * (xyz_x - xyz_y)
    lab_b = 200 * (xyz_y - xyz_z)

    return np.array([lab_l, lab_a, lab_b])


def lab2rgb_single(lab: np.ndarray):
    # convert a single pixel from lab to rgb
    # https://web.archive.org/web/20111111073514/http://www.easyrgb.com/index.php?X=MATH&H=08#text8
    assert lab.shape == (3,)
    lab_l, lab_a, lab_b = lab

    xyz_y = (lab_l + 16) / 116
    xyz_x = lab_a / 500 + xyz_y
    xyz_z = xyz_y - lab_b / 200

    f = lambda x: x ** 3 if x ** 3 > 0.008856 else (x - 16 / 116) / 7.787
    xyz_x = f(xyz_x)
    xyz_y = f(xyz_y)
    xyz_z = f(xyz_z)

    xyz_x *= 95.047
    xyz_y *= 100
    xyz_z *= 108.883

    xyz_x /= 100
    xyz_y /= 100
    xyz_z /= 100

    rgb_r = xyz_x * 3.2406 + xyz_y * -1.5372 + xyz_z * -0.4986
    rgb_g = xyz_x * -0.9689 + xyz_y * 1.8758 * xyz_z * 0.0415
    rgb_b = xyz_x * 0.0557 + xyz_y * -0.2040 + xyz_z * 1.0570

    g = lambda x: 1.055 * (x ** (1 / 2.4)) - 0.055 if x > 0.0031308 else 12.92 * x
    rgb_r = g(rgb_r)
    rgb_g = g(rgb_g)
    rgb_b = g(rgb_b)

    rgb_r *= 255
    rgb_g *= 255
    rgb_b *= 255

    return np.array([rgb_r, rgb_g, rgb_b])
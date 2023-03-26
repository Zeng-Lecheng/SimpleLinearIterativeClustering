import argparse
from PIL import Image
from slic import Slic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, help='path to the input')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='path to the output')
    parser.add_argument('-k', '--num-of-centroids', type=int, required=True, help='number of centroids')
    parser.add_argument('--max-iter', type=int, default=10, help='maximum number of iterations')
    parser.add_argument('-t', '--converge-threshold', type=float, default=0.01, help='if centroids move less than this '
                                                                                     'value after certain iteration, '
                                                                                     'then stop iterating')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    img = Image.open(args.input_path)
    model = Slic(img, args.num_of_centroids)
    model.fit(args.max_iter, args.converge_threshold)
    model.save(args.output_path)

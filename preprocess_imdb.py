import argparse
import glob
import math
import os
import sys

import cv2
from ml_serving.drivers import driver
import numpy as np

import hook


face_model_path = (
    '/opt/intel/openvino/deployment_tools/intel_models'
    '/face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--min-size', default=100, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    min_face_size = args.min_size
    min_box_diagonal = int(math.sqrt(2 * (min_face_size ** 2)))
    print_fun('List files...')
    image_paths = glob.glob(os.path.join(args.data_dir, '**/*.jpg'))
    print_fun(f'Done list files: {len(image_paths)}')

    print_fun('Load face detect driver...')
    face_driver = driver.load_driver('openvino')().load_model(face_model_path)
    print_fun('Done loading.')

    output_b = os.path.join(args.output_dir, 'trainB')
    output_a = os.path.join(args.output_dir, 'trainA')
    os.makedirs(output_a, exist_ok=True)
    os.makedirs(output_b, exist_ok=True)

    processed = 0
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print_fun(f'Progress {i / len(image_paths) * 100:.2f} %.')
            print_fun(f'Processed {processed} images, looked: {i}.')

        splitted = path.split('.')[-2].split('_')
        dob = splitted[-2]
        year = int(splitted[-1])
        birth_year = int(dob.split('-')[0])
        # print_fun(path, dob, year)
        # print_fun(birth_year)
        age = year - birth_year
        # print_fun(age)
        if 15 <= age <= 30:
            with open(path, 'rb') as f:
                raw_img = f.read()
            img = cv2.imdecode(np.frombuffer(raw_img, np.uint8), cv2.IMREAD_COLOR)
            save_path = os.path.join(output_a, os.path.basename(path))
        elif 50 <= age <= 70:
            with open(path, 'rb') as f:
                raw_img = f.read()
            img = cv2.imdecode(np.frombuffer(raw_img, np.uint8), cv2.IMREAD_COLOR)
            save_path = os.path.join(output_b, os.path.basename(path))
        else:
            continue

        if img.shape[0] * img.shape[1] < 40000:
            continue
        boxes = hook.get_boxes(face_driver, img, threshold=args.threshold)
        if len(boxes) != 1 or box_diagonal(boxes[0]) < min_box_diagonal:
            continue

        new_img = hook.crop_by_box(img, boxes[0])
        cv2.imwrite(save_path, new_img)
        processed += 1

        if args.limit != 0 and processed >= args.limit:
            break

    print_fun(f'Processed {processed} images.')


def print_fun(s):
    print(s)
    sys.stdout.flush()


def box_diagonal(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return math.sqrt(w ** 2 + h ** 2)


if __name__ == '__main__':
    main()

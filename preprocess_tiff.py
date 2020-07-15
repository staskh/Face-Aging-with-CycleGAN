import argparse
import glob
import os
import sys

import cv2
from ml_serving.drivers import driver

import hook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--face-model',
        default=('/opt/intel/openvino/deployment_tools/intel_models/'
                 'face-detection-adas-0001/FP32/face-detection-adas-0001.xml')
    )

    return parser.parse_args()


def main():
    args = parse_args()
    face_driver = driver.load_driver('openvino')()
    face_driver.load_model(args.face_model)

    train_a = sorted(glob.glob(os.path.join(args.data_dir, '*-1.tiff')))
    train_b = sorted(glob.glob(os.path.join(args.data_dir, '*-2.tiff')))

    output_a = os.path.join(args.output_dir, 'trainA')
    output_b = os.path.join(args.output_dir, 'trainB')
    os.makedirs(output_a, exist_ok=True)
    os.makedirs(output_b, exist_ok=True)

    print('Processing images...')
    for img_a_path, img_b_path in zip(train_a, train_b):
        img_a = cv2.imread(img_a_path)
        img_b = cv2.imread(img_b_path)
        base_a, _ = os.path.splitext(os.path.basename(img_a_path))
        base_b, _ = os.path.splitext(os.path.basename(img_b_path))

        boxes_a = hook.get_boxes(face_driver, img_a, threshold=0.2)
        boxes_b = hook.get_boxes(face_driver, img_b, threshold=0.2)

        if len(boxes_a) != 1 or len(boxes_b) != 1:
            print(f'Found {len(boxes_a)} boxes: {img_a_path}')
            print(f'Found {len(boxes_b)} boxes: {img_b_path}')
            continue

        img_a = hook.crop_by_box(img_a, boxes_a[0], margin=0.05)
        img_b = hook.crop_by_box(img_b, boxes_b[0], margin=0.05)

        cv2.imwrite(os.path.join(output_a, base_a + '.jpg'), img_a)
        cv2.imwrite(os.path.join(output_b, base_b + '.jpg'), img_b)
        print('.', end='')
        sys.stdout.flush()

    print()
    print(f'Done. Processed images are saved in {output_a} and {output_b}')


if __name__ == '__main__':
    main()

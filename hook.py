import json
import logging
import typing

import cv2
from ml_serving.utils import helpers
import numpy as np

import color_transfer


LOG = logging.getLogger(__name__)
PARAMS = {
    'threshold': 0.5,
    'color_transfer': False,
    'alpha': 255,
}


def init_hook(ctx, **params):
    PARAMS.update(params)
    PARAMS['alpha'] = int(PARAMS['alpha'])
    PARAMS['color_transfer'] = str2bool(PARAMS['color_transfer'])
    LOG.info('Init params:')
    new_params = {k: v for k, v in PARAMS.items() if k not in ctx.kwargs_for_hook()}
    LOG.info(json.dumps(new_params, indent=2))


def str2bool(v: typing.Union[bool, str]):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    return False


def get_param(inputs, key):
    value = inputs.get(key)
    if value is None:
        return PARAMS.get(key)

    if hasattr(value, 'shape'):
        if len(value.shape) == 1:
            value = value[0]

        if len(value.shape) == 0:
            value = value.tolist()

    if isinstance(value, bytes):
        value = value.decode()

    return value


def process(inputs, ctx):
    if len(ctx.drivers) < 2:
        raise RuntimeError('Required 2 models: face and cyclegan')

    enable_color_transfer = get_param(inputs, 'color_transfer')
    alpha = get_param(inputs, 'alpha')

    face_driver = ctx.drivers[0]
    cyclegan_driver = ctx.drivers[1]
    input_name = list(cyclegan_driver.inputs.keys())[0]

    original, is_video = helpers.load_image(inputs, 'input')
    image = original.copy()

    boxes = get_boxes(face_driver, image)
    for box in boxes:
        box = box.astype(int)
        img = crop_by_box(image, box)

        prepared = np.expand_dims(prepare_image(img), axis=0)
        outputs = cyclegan_driver.predict({input_name: prepared})
        output = list(outputs.values())[0].squeeze()
        output = inverse_transform(output)
        output = scale(output)
        # output = (output * 255).astype(np.uint8)
        output = cv2.resize(np.array(output), (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_AREA)

        if enable_color_transfer:
            output = color_transfer.color_transfer(img, output, clip=True, preserve_paper=False)

        center = (box[0] + output.shape[1] // 2, box[1] + output.shape[0] // 2)
        alpha = np.clip(alpha, 1, 255)
        image = cv2.seamlessClone(output, image, np.ones_like(output) * alpha, center, cv2.NORMAL_CLONE)

        # image[box[1]:box[3], box[0]:box[2]] = (output / 2 + img / 2).astype(np.uint8)

    # merge
    image = np.vstack((original, image))

    if not is_video:
        image = image[:, :, ::-1]
        image_bytes = cv2.imencode('.jpg', image)[1].tostring()
    else:
        image_bytes = image

    return {'output': image_bytes}


def scale(img, high=255, low=0, cmin=None, cmax=None):
    if not cmin:
        cmin = img.min()
        cmax= img.max()

    cscale = cmax - cmin
    scale = float(high - low) / cscale
    bytedata = (img - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def inverse_transform(images):
    return (images + 1.) / 2.


def prepare_image(img):
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img / 127.5 - 1.


def get_boxes(face_driver, frame, threshold=0.5, offset=(0, 0)):
    input_name, input_shape = list(face_driver.inputs.items())[0]
    output_name = list(face_driver.outputs)[0]
    inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = face_driver.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    xmin = boxes[:, 0] * frame.shape[1] + offset[0]
    xmax = boxes[:, 2] * frame.shape[1] + offset[0]
    ymin = boxes[:, 1] * frame.shape[0] + offset[1]
    ymax = boxes[:, 3] * frame.shape[0] + offset[1]
    xmin[xmin < 0] = 0
    xmax[xmax > frame.shape[1]] = frame.shape[1]
    ymin[ymin < 0] = 0
    ymax[ymax > frame.shape[0]] = frame.shape[0]

    boxes[:, 0] = xmin
    boxes[:, 2] = xmax
    boxes[:, 1] = ymin
    boxes[:, 3] = ymax
    return boxes


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box, margin=0.):
    h = (box[3] - box[1])
    w = (box[2] - box[0])
    ymin = int(max([box[1] - h * margin, 0]))
    ymax = int(min([box[3] + h * margin, img.shape[0]]))
    xmin = int(max([box[0] - w * margin, 0]))
    xmax = int(min([box[2] + w * margin, img.shape[1]]))
    return img[ymin:ymax, xmin:xmax]

import logging
import time
from ml_serving.drivers import driver

import cv2
from ml_serving.utils import helpers
import numpy as np

LOG = logging.getLogger(__name__)


class TFOpenCVFaces:
    def __init__(self, model_path):
        self._model_path = model_path
        drv = driver.load_driver('tensorflow')
        self.serving = drv()
        constants0 = {'data_bn/beta:0': 3, 'conv1_bn_h/beta:0': 32, 'layer_128_1_bn1_h/beta:0': 32,
                      'layer_256_1_bn1/beta:0': 128, 'layer_512_1_bn1/beta:0': 256, 'last_bn_h/beta:0': 256}
        input_names = ['data:0']
        self.inputs = {}
        for k, v in constants0.items():
            input_names.append(k)
            self.inputs[k] = np.zeros((v), np.float32)
        self.serving.load_model(self._model_path + '/opencv_face_detector_uint8.pb', inputs=','.join(input_names),
                                outputs='mbox_loc:0,mbox_conf_flatten:0')
        configFile = self._model_path + "/detector.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(None, configFile)

        self.prior = np.fromfile(self._model_path + '/mbox_priorbox.np', np.float32)
        self.prior = np.reshape(self.prior, (1, 2, 35568))
        self.threshold = 0.5

    def bboxes(self, frame):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame[:, :, ::-1], 1.0, (300, 300), [104, 117, 123], False, False)
        blob = np.transpose(blob, (0, 2, 3, 1))
        self.inputs['data:0'] = blob
        result = self.serving.predict(self.inputs)
        probs = result.get('mbox_conf_flatten:0')
        boxes = result.get('mbox_loc:0')

        self.net.setInput(boxes, name='mbox_loc')
        self.net.setInput(probs, name='mbox_conf_flatten')
        self.net.setInput(self.prior, name='mbox_priorbox')
        detections = self.net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)
                y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)
                x2 = min(int(detections[0, 0, i, 5] * frameWidth), frameWidth)
                y2 = min(int(detections[0, 0, i, 6] * frameHeight), frameHeight)
                if x1 > x2:
                    x2 = x1
                if y1 > y2:
                    y2 = y1
                bboxes.append((np.array([x1, y1, x2, y2], np.int32), confidence))
        return bboxes

    def stop(self, ctx):
        del self.serving


class Pipe:
    def __init__(self, ctx, **params):
        self._alpha = int(params.get('alpha', 255))
        self._draw_box = srt_2_bool(params.get('draw_box', False))
        self._face_detection_path = params.get('face_detection_path', None)
        self._style_size = int(params.get('style_size', 256))
        self._face_detection_type = params.get('face_detection_type', None)
        if self._face_detection_type == 'tf-opencv':
            self.face_detector = TFOpenCVFaces(self._face_detection_path)
        self._output_view = params.get('output_view', 'split_horizontal')
        self._transfer_mode = params.get('transfer_mode', 'box_margin')
        self._style_driver = ctx.drivers[0]
        self._style_input_name = list(self._style_driver.inputs.keys())[0]

    def get_param(self, inputs, key, def_val=None):
        value = inputs.get(key)
        if value is None:
            return def_val
        if hasattr(value, 'shape'):
            if len(value.shape) == 1:
                value = value[0]

            if len(value.shape) == 0:
                value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode()

        return value

    def process(self, inputs, ctx):
        alpha = int(self.get_param(inputs, 'alpha', self._alpha))
        style_size = int(self.get_param(inputs, 'style_size', self._style_size))
        original, is_video = helpers.load_image(inputs, 'input')
        image = original.copy()
        st1 = time.time()
        boxes = self.face_detector.bboxes(image)
        LOG.info('BoxTime: {}ms'.format(int((time.time() - st1) * 1000)))
        for box, confidence in boxes:
            box = box.astype(int)
            img = image[box[1]:box[3], box[0]:box[2],:]
            prepared = np.expand_dims(prepare_image(img, style_size), axis=0)
            st1 = time.time()
            outputs = self._style_driver.predict({self._style_input_name: prepared})
            LOG.info('StyleTime: {}ms'.format(int((time.time() - st1) * 1000)))
            output = list(outputs.values())[0].squeeze()
            output = inverse_transform(output)
            output = scale(output)
            alpha = np.clip(alpha, 1, 255)
            if self.get_param(inputs, 'transfer_mode', self._transfer_mode) == 'color_transfer':
                st1 = time.time()
                output = color_tranfer(output, img)
                output = cv2.resize(output, (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_AREA)
                image[box[1]:box[3], box[0]:box[2], :] = output
                LOG.info('ColorTransfer: {}ms'.format(int((time.time() - st1) * 1000)))
            else:
                output = cv2.resize(np.array(output), (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_AREA)
                if self.get_param(inputs, 'transfer_mode', self._transfer_mode) == 'box_margin':
                    xmin = max(0, box[0] - 50)
                    wleft = box[0] - xmin
                    ymin = max(0, box[1] - 50)
                    wup = box[1] - ymin
                    xmax = min(image.shape[1], box[2] + 50)
                    # wright = xmax-box[2]
                    ymax = min(image.shape[0], box[3] + 50)
                    # wdown = ymax-box[3]
                    out = image[ymin:ymax, xmin:xmax, :]
                    center = (wleft + output.shape[1] // 2, wup + output.shape[0] // 2)
                    st1 = time.time()
                    out = cv2.seamlessClone(output, out, np.ones_like(output) * alpha, center, cv2.NORMAL_CLONE)
                    LOG.info('CloneTime Box: {}ms'.format(int((time.time() - st1) * 1000)))
                    image[ymin:ymax, xmin:xmax, :] = out
                else:
                    center = (box[0] + output.shape[1] // 2, box[1] + output.shape[0] // 2)
                    st1 = time.time()
                    image = cv2.seamlessClone(output, image, np.ones_like(output) * alpha, center, cv2.NORMAL_CLONE)
                    LOG.info('CloneTime Full: {}ms'.format(int((time.time() - st1) * 1000)))
            if len(box) > 0:
                if srt_2_bool(self.get_param(inputs, "draw_box", self._draw_box)):
                    LOG.info('DrawBox: {}'.format(srt_2_bool(self.get_param(inputs, "draw_box", self._draw_box))))
                    image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 8)
                    break
        # merge
        output_view = self.get_param(inputs, 'output_view', self._output_view)
        if output_view == 'split_horizontal':
            image = np.hstack((original, image))
        elif output_view == 'split_vertical':
            image = np.vstack((original, image))
        if not is_video:
            image = image[:, :, ::-1]
            image_bytes = cv2.imencode('.jpg', image)[1].tostring()
        else:
            image_bytes = image

        return {'output': image_bytes}

    def stop(self, ctx):
        self.face_detector.stop(ctx)


def init_hook(ctx, **params):
    LOG.info('Init params:')
    return Pipe(ctx, **params)


def update_hook(ctx, **kwargs):
    if ctx.global_ctx is not None:
        LOG.info('close existing pipe')
        ctx.global_ctx.stop(ctx)
    return Pipe(ctx, **kwargs)


def process(inputs, ctx):
    return ctx.global_ctx.process(inputs, ctx)


def scale(img, high=255, low=0, cmin=None, cmax=None):
    if not cmin:
        cmin = img.min()
        cmax = img.max()

    cscale = cmax - cmin
    scale = float(high - low) / cscale
    bytedata = (img - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def inverse_transform(images):
    return (images + 1.) / 2.


def prepare_image(img, style_size=256):
    img = cv2.resize(img, (style_size, style_size), interpolation=cv2.INTER_AREA)
    return img / 127.5 - 1.



def srt_2_bool(v):
    if v == 'True':
        return True
    if v == 'False':
        return False
    return bool(v)


def color_tranfer(s, t):
    s = cv2.cvtColor(s, cv2.COLOR_RGB2LAB)
    t = cv2.cvtColor(t, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)
    s = ((s-s_mean)*(t_std/s_std))+t_mean
    s = np.clip(s,0,255).astype(np.uint8)
    return cv2.cvtColor(s, cv2.COLOR_LAB2RGB)


def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

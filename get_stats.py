import argparse

from prometheus_client import parser
import requests
import time
from curses import wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--endpoint',
        default='http://localhost:9090/metrics'
    )

    return parser.parse_args()


args = None


def get_model_time(metrics_dict):
    vals = {}
    for m in metrics_dict['serving_process_request_ms_avg_time']:
        mpath = m.labels.get('model_path', '')
        if 'face-detection' in mpath:
            vals['face-detection'] = m.value
        elif 'head-pose' in mpath:
            vals['head-pose'] = m.value
        elif 'facenet' in mpath:
            vals['face-recognition'] = m.value
    return vals


def stats(stdscr):
    while True:
        stdscr.clear()
        metrics_raw = requests.get(args.endpoint).content.decode()
        metrics = parser.text_string_to_metric_families(metrics_raw)
        metrics_dict = {}
        for family in metrics:
            if family.name in metrics_dict:
                metrics_dict[family.name].extend(family.samples)
            else:
                metrics_dict[family.name] = family.samples

        row = 0
        values = get_model_time(metrics_dict)
        stdscr.addstr(row, 0,
                      'Detection average time: {:.3f}ms'.format(values.get('face-detection', -1)))
        row += 1
        stdscr.addstr(row, 0,
                      'Pose-Estimation average time: {:.3f}ms'.format(values.get('head-pose', -1)))
        row += 1
        stdscr.addstr(row, 0,
                      'Recognition average time: {:.3f}ms'.format(values.get('face-recognition', -1)))
        row += 1

        stdscr.addstr(row, 0,
                      'Inference average time: {:.3f}ms'.format(metrics_dict['serving_request_ms_avg_time'][0].value))
        row += 1
        rtime = metrics_dict['serving_request_time'][0].value
        fps = 1 / rtime * 1000 if rtime != 0 else 0
        stdscr.addstr(row, 0,
                      'Max Inference FPS: {:.2f}'.format(fps))
        stdscr.refresh()
        time.sleep(5)


def main():
    global args
    args = parse_args()
    wrapper(stats)


if __name__ == '__main__':
    main()

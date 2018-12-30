#! /usr/bin/python
import sys
import getopt
import cv2
import numpy as np
import random
from recordclass import recordclass

State = recordclass('State', ['x', 'y', 'vx', 'vy'])


def print_usage():
    print("Usage: ./video_generator.py -o <output file> [OPTIONS]")


def generate_init_states(num_objects, vx, vy, resolution):
    states = []
    for i in range(num_objects):
        x = int(resolution[0]*random.random())
        y = int(resolution[1]*random.random())
        direction = -1 if random.random() < 0.5 else 1
        choices = [
            State(x, 0, direction*vx, vy),                 # top edge
            State(x, resolution[1]-1, direction*vx, -vy),  # bottom edge
            State(0, y, vx, direction*vy),                 # left edge
            State(resolution[0]-1, y, -vx, direction*vy)   # right edge
        ]
        states.append(random.choice(choices))

    return states


def is_outside_frame_bounds(point, resolution):
    if point[0] > resolution[0] \
            or point[0] < 0 \
            or point[1] > resolution[1] \
            or point[1] < 0:
        return True
    return False


def main(argv):
    try:
        opts, args = getopt.getopt(
            argv,
            "ho:l:x:y:r:n:b:",
            ["help", "outfile=", "loops=", "hres=", "yres=", "radius=", "velx=", "vely=",
             "noise-factor=", "num-objects="]
        )
    except getopt.GetoptError:
        print_usage()
        return

    outfile = None
    num_loops = 1
    h_res = 1024
    v_res = 768
    vx = 5.0
    vy = 5.0
    radius = 15
    noise_factor = 0
    num_objects = 1

    for opt, arg in opts:
        if opt=='-h' or opt=='--help':
            print_usage()
        elif opt=='-o' or opt=='--outfile':
            outfile = arg
        elif opt=='-l' or opt=='--loops':
            num_loops = int(arg)
        elif opt=='-x' or opt=='--hres':
            h_res = int(arg)
        elif opt=='-y' or opt=='--vres':
            v_res = int(arg)
        elif opt=='-r' or opt=='--radius':
            radius = float(opt)
        elif opt=='-n' or opt=='--noise-factor':
            noise_factor = float(arg)
        elif opt=='-b' or opt=='--num-objects':
            num_objects = int(arg)
        elif opt == '--velx':
            vx = float(opt)
        elif opt == '--vely':
            vy = float(opt)

    if outfile:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vid = cv2.VideoWriter(outfile, fourcc, 20.0, (h_res, v_res))
    else:
        print("No output file provided.")
        print_usage()
        return

    for _ in range(num_loops):
        states = generate_init_states(num_objects, vx, vy, (h_res, v_res))
        while states:
            frame = np.zeros((v_res, h_res, 3), dtype=np.uint8)
            tmp = []
            for i in range(len(states)):
                cv2.circle(frame, (int(states[i].x), int(states[i].y)),
                           radius, (255, 255, 255), thickness=-1)
                states[i].x += states[i].vx + random.uniform(-noise_factor, noise_factor)
                states[i].y += states[i].vy + random.uniform(-noise_factor, noise_factor)
                if not is_outside_frame_bounds((states[i].x, states[i].y), (h_res, v_res)):
                    tmp.append(states[i])
            states = tmp
            vid.write(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    vid.release()


if __name__ == "__main__":
    main(sys.argv[1:])

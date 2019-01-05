#! /usr/bin/python
import sys
import getopt
import cv2
import numpy as np
import random
from recordclass import recordclass

State = recordclass('State', ['x', 'y', 'vx', 'vy'])

class VideoGenerator:
    def __init__(self, hres, vres, vx, vy, noise_factor, num_objects, radius):
        self.resolution = (hres, vres)
        self.vx = vx
        self.vy = vy
        self.noise_factor = noise_factor
        self.num_objects = num_objects
        self.radius = radius
        self.states = []

    def print_usage():
        print("Usage: ./video_generator.py -o <output file> [OPTIONS]")
    
    
    def generate_init_states(self):
        states = []
        for i in range(self.num_objects):
            x = int(self.resolution[0]*random.random())
            y = int(self.resolution[1]*random.random())
            direction = -1 if random.random() < 0.5 else 1
            choices = [
                State(x, 0, direction*self.vx, self.vy),                      # top edge
                State(x, self.resolution[1]-1, direction*self.vx, -self.vy),  # bottom edge
                State(0, y, self.vx, direction*self.vy),                      # left edge
                State(self.resolution[0]-1, y, -self.vx, direction*self.vy)   # right edge
            ]
            states.append(random.choice(choices))
    
        self.states = states

    def generate_frame(self):
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        tmp = []
        if len(self.states) == 0:
            self.generate_init_states()

        for i in range(len(self.states)):
            cv2.circle(frame, (int(self.states[i].x), int(self.states[i].y)),
                       self.radius, (255, 255, 255), thickness=-1)
            self.states[i].x += self.states[i].vx + random.uniform(-self.noise_factor, self.noise_factor)
            self.states[i].y += self.states[i].vy + random.uniform(-self.noise_factor, self.noise_factor)
            if not VideoGenerator.is_outside_frame_bounds((self.states[i].x, self.states[i].y), self.resolution):
                tmp.append(self.states[i])
        self.states = tmp
        return frame

    
    @staticmethod
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

    generator = VideoGenerator(h_res, v_res, vx, vy, noise_factor, num_objects, radius)
    for _ in range(num_loops):
        while True:
            frame = generator.generate_frame()
            vid.write(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            if not generator.states:
                break

    vid.release()


if __name__ == "__main__":
    main(sys.argv[1:])

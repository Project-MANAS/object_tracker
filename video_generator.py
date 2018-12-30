import sys
import getopt
import cv2
import numpy as np
import random


def print_usage():
    print("Usage: ./video_generator.py -o <output file>")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "ho:l:x:y:", ["help", "outfile=", "loops=", "hres=", "yres="])
    except getopt.GetoptError:
        print_usage()

    outfile = None
    num_loops = 1
    h_res = 1024
    v_res = 768
    vx_max = 0.5
    vy_max = 0.5
    radius = 5

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
        else:
            print("Unknown option "+opt)
            print_usage()

    if outfile:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vid = cv2.VideoWriter(outfile, fourcc, 20.0, (h_res, v_res))
    else:
        print("No output file provided.")
        print_usage()
        return

    for _ in range(num_loops):
        x = 0
        y = random.random()*v_res
        vx = random.random()*vx_max
        vy = random.random()*vy_max
        while x<=h_res and y<=v_res:
            frame = np.zeros((v_res, h_res, 3), dtype=np.uint8)
            cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), thickness=-1)
            vid.write(frame)
            x += vx
            y += vy
            #cv2.imshow('frame', frame)
            #cv2.waitKey(1)

    vid.release()


if __name__ == "__main__":
    main(sys.argv[1:])

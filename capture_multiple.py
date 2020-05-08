import gphoto2 as gp
import os 
import time
import cv2
import numpy as np


class DSLR():

    def __init__(self, dir):

        dir = os.path.abspath(args.dir)
        self.context = gp.Context()
        self.camera = gp.Camera()
        self.camera.init(context)
        self.path='{}/{}'.format(dir, time.strftime('%S_%M_%H_%d_%m_%Y',time.localtime()))
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
    
    def capture(self, nImages, settype):
        cameradata=[]
        files=[]
        for i in range(args.nImages):
            print('Capturing image')
            file_path = camera.capture(gp.GP_CAPTURE_IMAGE, context)
            camera.wait_for_event(gp.GP_EVENT_CAPTURE_COMPLETE, context)
            print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
            target = os.path.join(self.path, settype, file_path.name)

            if os.path.exists('{}/{}'.format(self.path, settype)):
                os.mkdir(target)

            files.append(target)
            camera_file = gp.check_result(camera.file_get(
                file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL, context))
        cameradata.append([camera_file, target])
        for camera_file in cameradata:
            print('Copying image to', camera_file[1])
            camera_file[0].save(camera_file[1])
    
    def __del__(self):
        camera.exit(context)
        print("Context cleaaared")

def parse_args():
    parser=argparse.ArgumentParser(description='Stack a series of images captured by a camera of the night sky')
    parser.add_argument('-d', '--dir', dest='dir', help='directory to the images from a run')
    parser.add_argument('-n', dest='nImages', type=int, help='Number of images to capture this time')
    parser.add_argument('--light', dest='light', action='store_true', help='Save images as a light set')
    parser.add_argument('--dark', dest='dark', action='store_true', help='Save images as a dark set: Ie same camera settings including temperature with the lens cover on')
    parser.add_argument('--flat', dest='flat', action='store_true',  help='Save images as flat set: ie Evenly Illuminated target through the optics')
    parser.add_argument('--darkflat', dest='darkflat', action='store_true',  help='Save images as dark flat set: ie Evenly Dark Illuminated target through the optics')
    parser.add_argument('--bias', dest='bias', action='store_true',  help='Save images as flat set: ie shortest exposure possible at same camera settings')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    cam = DSLR(args.dir)
    if args.light:
        cam.capture(args.nImages, 'light')
    elif args.dark:
        cam.capture(args.nImages, 'dark')
    elif args.flat:
        cam.capture(args.nImages, 'flat')
    elif args.bias:
        cam.capture(args.nImages, 'bias')
    elif args.darkflat:
        cam.capture(args.nImages, 'darkflat')
    else
        cam.capture(args.nImages, 'light')

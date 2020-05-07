import gphoto2 as gp
import os 
import time
import cv2
import numpy as np

def parse_args():
    parser=argparse.ArgumentParser(description='Stack a series of images captured by a camera of the night sky')
    parser.add_argument('-d', '--dir', dest='dir', help='directory to the images from a run')
    parser.add_argument('-n', dest='nImages', type=int help='Number of images to capture this time')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args() 
    dir = os.path.abspath(args.dir)
    context = gp.Context()
    camera = gp.Camera()
    camera.init(context)
    files=[]
    path='{}/{}'.format(dir, time.strftime('%S_%M_%H_%d_%m_%Y',time.localtime()))
    if not os.path.exists(path):
            os.mkdir(path)
    os.chdir(path)
    cameradata=[]
    for i in range(args.nImages):
        print('Capturing image')
        file_path = camera.capture(gp.GP_CAPTURE_IMAGE, context)
        camera.wait_for_event(gp.GP_EVENT_CAPTURE_COMPLETE, context)
        print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
        target = os.path.join(os.getcwd(), file_path.name)
        files.append(target)
        camera_file = gp.check_result(camera.file_get(
                file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL, context))
        #file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
        cameradata.append([camera_file, target])
    for camera_file in cameradata:
        print('Copying image to', camera_file[1])
        camera_file[0].save(camera_file[1])

        #img = cv2.imread(target)
        #cv2.imshow('Captured_Image', img)
        #cv2.waitKey(500)
    camera.exit(context)
    print("Context cleaaared")

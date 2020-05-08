
import numpy as np
import opencv as cv2

class AstroCalibration():

    def __init__(self, rootdir):
	self.rootdir = rootdir
    
    def _checkForData(self):
	
        folder_present=True
	folder_present &= os.path.exists('{}/light'.format(self.rootdir))
	folder_present &= os.path.exists('{}/bias'.format(self.rootdir))
	folder_present &= os.path.exists('{}/dark'.format(self.rootdir))
	folder_present &= os.path.exists('{}/flat'.format(self.rootdir))
	return folder_present

    def makeMasterBiasFrame(self):
        #TODO combine all bias frames via the per pixel median for all frames
	a=0
    def makeMasterDarkFrame(self):
	a=0
	#TODO subtract bias frame from each dark frame
	#TODO average all the dark frames

    def makeMasterDarkFlatFrame(self):
	a=0
	#TODO subtract master bias frame from each dark flat frames
	#TODO average all the dark flat frames

    def makeMasterFlatFieldFrame(self):
	a=0
	#TODO subtract master bias frame from each flat field frame
	#TODO subtract master dark flat frame from each flat field frame
	#TODO Average all the flat field frames

    def calibrateLightFrames(self):
	a=0
	#TODO subtract master bias frame
	#TODO subtract master dark frame
	#TODO divide by MAster Flat Field 
    
    def getImagesInFolder(self):
	#TODO return a list of images existing in a folder. As the folder is the way we do it
	a=0
if __name__ == '__main__':
    a=0


import numpy as np
import cv2
import os

class AstroCalibration():

    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.masterBiasFrame =None
        self.masterDarkFrame=None
        self.masterDarkFlatFrame=None
        self.masterFlatFrame=None
        
    def _checkForData(self):
        
        folder_present=True
        folder_present &= os.path.exists('{}/light'.format(self.rootdir))
        folder_present &= os.path.exists('{}/bias'.format(self.rootdir))
        folder_present &= os.path.exists('{}/dark'.format(self.rootdir))
        folder_present &= os.path.exists('{}/darkflat'.format(self.rootdir))
        folder_present &= os.path.exists('{}/flat'.format(self.rootdir))
        return folder_present

    def makeMasterBiasFrame(self):
        #TODO combine all bias frames via the per pixel median for all frames
        images = self.getImagesInFolder(os.path.join(self.rootdir, 'bias'))
        self.masterBiasFrame = np.zeros(images[0].shape)

        r = np.stack([x[:,:,0] for x in images], axis=2)
        g = np.stack([x[:,:,1] for x in images], axis=2)
        b = np.stack([x[:,:,2] for x in images], axis=2)
        
        assert(r.shape[2] == len(images))

        self.masterBiasFrame[:,:,0] = np.median(r, axis=2)
        self.masterBiasFrame[:,:,1] = np.median(g, axis=2)
        self.masterBiasFrame[:,:,2] = np.median(b, axis=2)
        
        assert(self.masterBiasFrame.shape == images[0].shape)

        return self.masterBiasFrame


    def makeMasterDarkFrame(self):
        a=0
        #TODO subtract bias frame from each dark frame
        #TODO average all the dark frames
        if self.masterBiasFrame is not None:
            
            images = self.getImagesInFolder(os.path.join(self.rootdir, 'dark'))
            self.masterDarkFrame = np.zeros(images[0].shape)

            r = np.stack([x[:,:,0] for x in images], axis=2)
            g = np.stack([x[:,:,1] for x in images], axis=2)
            b = np.stack([x[:,:,2] for x in images], axis=2)
        
            assert(r.shape[2] == len(images))
            
            r[:,:,:] -= self.masterBiasFrame[:,:,0]
            g[:,:,:] -= self.masterBiasFrame[:,:,1]
            b[:,:,:] -= self.masterBiasFrame[:,:,2]

            self.masterDarkFrame[:,:,0] = np.mean(r, axis=2)
            self.masterDarkFrame[:,:,1] = np.mean(g, axis=2)
            self.masterDarkFrame[:,:,2] = np.mean(b, axis=2)
        
            assert(self.masterDarkFrame.shape == images[0].shape)

    def makeMasterDarkFlatFrame(self):
        a=0
        #TODO subtract master bias frame from each dark flat frames
        #TODO average all the dark flat frames

        if self.masterBiasFrame is not None:
            
            images = self.getImagesInFolder(os.path.join(self.rootdir, 'darkflat'))
            self.masterDarkFlatFrame = np.zeros(images[0].shape)

            r = np.stack([x[:,:,0] for x in images], axis=2)
            g = np.stack([x[:,:,1] for x in images], axis=2)
            b = np.stack([x[:,:,2] for x in images], axis=2)
        
            assert(r.shape[2] == len(images))
            
            r[:,:,:] -= self.masterBiasFrame[:,:,0]
            g[:,:,:] -= self.masterBiasFrame[:,:,1]
            b[:,:,:] -= self.masterBiasFrame[:,:,2]

            self.masterDarkFlatFrame[:,:,0] = np.mean(r, axis=2)
            self.masterDarkFlatFrame[:,:,1] = np.mean(g, axis=2)
            self.masterDarkFlatFrame[:,:,2] = np.mean(b, axis=2)
        
            assert(self.masterDarkFrame.shape == images[0].shape)

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
    
    def getImagesInFolder(self, dir):
        #TODO return a list of images existing in a folder. As the folder is the way we do it
        imagelist=[]
        contents =os.listdir(dir)
        for cont in contents:
            if '.jpg' in cont:
                im = cv2.imread('{}/{}'.format(dir, cont))
                #print(im)
                imagelist.append(im)
        
        return imagelist
        
if __name__ == '__main__':
    
    ac = AstroCalibration('/home/peter/Pictures/astro/test/26_18_20_11_09_2019')
    ac.makeMasterBiasFrame()


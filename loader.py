import glob
import numpy as np 
import scipy as scipy

class loader:
	def __init__(self):
		self.depth_folder = '../DepthMap_dataset-master/Depth_map'
		self.im_folder = '../DepthMap_dataset-master/StereoImages'
		self.fileList = glob.glob(self.depth_folder + '/*.png');
		self.cube_len = 64;

	def getBatch(self, batchSize):

		#create an appropriately sized numpy array
		batch = np.zeros((batchSize,height,width,2), dtype=np.float32)
		y = np.zeros((batchSize,height,width,1), dtype=np.float32)


		idx = np.random.permutation(len(self.fileList));
		idx = idx[0:batchSize-1];

		data = {}
		#grab a random set of them

		for i in range(0,len(idx)):

			depth_file = self.fileList[idx[i]]
			im_file = depth_file.strrep('Depth','Stereoscopic')
			print('depth file: ' + depth_file)
			print('im file: ' + im_file)

			depth = scipy.misc.imread(depth_file)
			im = scipy.misc.imread(im_file,mode='L')

			dims = np.shape(im);
			half = int(dims[1]/2);
			im_left = im[:,0:half-1];
			im_left = scipy.misc.imresize(im_left,[height,width])
			im_right = im[:,half:dims[2]-1];
			im_right = scipy.misc.imresize(im_right,[height,width])
			batch[i,:,:,0] = im_left;
			batch[i,:,:,1] = im_right;
			y[i,:,:] = depth;

		data['x'] = batch;
		data['y'] = y;

		return data;
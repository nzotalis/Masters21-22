from glob import glob
import cv2, skimage, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]

        self.img = {}
        self.R = np.eye(3)
        self.t = np.zeros(shape=(3,1))
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def detect(self, img):
        detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        p0 = detector.detect(img)
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        predictions = []
        predictions.append(np.zeros(shape=3))

        for i in tqdm(range(1, len(self.pose))):
            image1 = self.imread(self.frames[i - 1])
            image2 = self.imread(self.frames[i])

            self.p0 = self.detect(image1)

            self.p1, st, _ = cv2.calcOpticalFlowPyrLK(image1, image2, self.p0, None)

            q1 = self.p0[st == 1]
            q2 = self.p1[st == 1]

            E, _ = cv2.findEssentialMat(q2, q1, focal = self.focal_length, pp = self.pp)
            _, R, t, _ = cv2.recoverPose(E, q1, q2, focal = self.focal_length, pp = self.pp)

            scale = self.get_scale(i)
            self.t += scale * (self.R @ t).reshape(3,1)
            self.R = self.R @ R

            diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
            adj_coord = np.matmul(diag, self.t)

            predictions.append(adj_coord.squeeze())

        return np.array(predictions)

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    np.save('predictions.npy', path)
    gt_path = np.array([odemotryc.get_gt(i).squeeze() for i in range(len(path))])
    
    ax = plt.axes(projection='3d')
    ax.plot3D(gt_path[:,0], gt_path[:,1], gt_path[:,2], color="green")
    ax.plot3D(path[:,0], path[:,1], path[:,2], color="red")
    plt.show()
    


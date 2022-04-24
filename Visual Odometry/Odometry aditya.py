from glob import glob
import cv2, skimage, os
import numpy as np
#print(cv2.__version__)

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
        
        """MY CHANGES START"""
        
        self.K = np.array([[self.focal_length,0,self.pp[0]],[0,self.focal_length,self.pp[1]],[0,0,1]])
        
        self.frame1_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        
        self.D = np.array([[0,1,0],[-1,0,0],[0,0,1]])
                
        self.fast = cv2.FastFeatureDetector_create(threshold=25,nonmaxSuppression=True)

        """MY CHANGES END"""
        
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
    
    """-------------------------------------------------MY FUNCTIONS START-------------------------------------------------"""
    
    def check_E(self,E):
        
        _,S,V = np.linalg.svd(E)
        print(S)
        
        check = E@np.transpose(E)@E - (1.0/2)*np.trace(E@np.transpose(E))*E
        
        return check
    
    def check_F(self,F):
        return np.linalg.det(F)
    
    def get_matches_FAST_optical_flow(self,img1,img2):
   
        p_fast = self.fast.detect(img1)
        
        p0 = []
        for i in p_fast:
            p0.append(i.pt)
        
        p0 = np.array(p0,dtype=np.float32).reshape(-1,1,2)
                
        p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)
     
        points_1 = p0[st==1]
        points_2 = p1[st==1]
        
        return np.array(points_1),np.array(points_2)
    
    def get_matches_ORB(self,img1,img2):
        
        orb = cv2.ORB_create()
        
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        matches = bf.match(des1,des2)
        
        matches = sorted(matches, key = lambda x:x.distance)
        
        points_1 = []
        points_2 = []
        
        for m in matches:
            points_1.append(kp1[m.queryIdx].pt)
            points_2.append(kp2[m.trainIdx].pt)
        
        return np.array(points_1),np.array(points_2)
    
    def get_F (self,pts1,pts2):
        
        A = []
        for i in range(pts1.shape[0]):
            
            x = pts1[i][0]
            x_2 = pts2[i][0]
            y = pts1[i][1]
            y_2 = pts2[i][1]
            
            A.append([x_2*x, x_2*y, x_2, y_2*x, y_2*y, y_2, x, y, 1])
        
        A = np.array(A)
        
        _,_,V = np.linalg.svd(A)
        
        F = np.reshape(V[-1,:],(3,3))
        F = F/F[-1,-1]
        
        U,S,V = np.linalg.svd(F)
        
        S[-1]=0
        
        F = U @ np.diag(S) @ V
        
        return F
    
    def get_E (self,pts1,pts2,K):
        
        E,_ = cv2.findEssentialMat(pts2, pts1,K)
        
        """
        F=self.get_F(pts2,pts1)
        E = K.T @ F @ K             #E = np.dot(np.dot(K.T,F),K)
        #print(self.check_F(F))
        #print(self.check_E(E))
        #"""        
        
        return E
    
    def get_R_t (self, E, pts1, pts2, K):
        
        _,R,t,_ = cv2.recoverPose(E,pts2, pts1,K)
  
        """
        U,S,V = np.linalg.svd(E)
        
        t = U[:,-1].reshape(3,1)
        R = U @ self.D @ V
        #"""
    
        return R,t
    
    def update_R_t (self, pose_old, R, t, frame_no):
        
        R_old = pose_old[:3,:3]
        t_old = pose_old[:,3].reshape(3,1)

        t_new = self.get_scale(frame_no)*(R_old @ t) + t_old

        R_new = R_old @ R 

        pose_new = np.hstack([R_new,t_new.reshape(3,1)])

        return pose_new

    
    """-------------------------------------------------MY FUNCTIONS END-------------------------------------------------"""
    
    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        

        prediction = np.zeros((len(self.pose),3))
        
        prediction[0,:] = np.array([self.frame1_pose[0,3],self.frame1_pose[1,3],self.frame1_pose[2,3]])
         
        """
        mx,my=self.imread(self.frames[0]).shape
        scaled_K = np.array([[1.0/mx,0,0],[0,1.0/my,0],[0,0,1]])@self.K
        #"""
        
        for i in range(1, len(self.frames)):
            img1=self.imread(self.frames[i-1])
            img2=self.imread(self.frames[i])
            
            #match_pts1, match_pts2 = self.get_matches_ORB(img1,img2)
            
            match_pts1, match_pts2 = self.get_matches_FAST_optical_flow(img1, img2)

            E = self.get_E(match_pts1, match_pts2,self.K)

            R,t = self.get_R_t(E,match_pts1, match_pts2,self.K)
            
            new_pose = self.update_R_t(self.frame1_pose, R,t, i)
            
            prediction[i,:] = np.array([new_pose[0,3],new_pose[1,3],new_pose[2,3]])
             
            self.frame1_pose = new_pose
            
            print(i,end="\r")

        np.save('predictions.npy',prediction)
        
        return prediction
        

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)

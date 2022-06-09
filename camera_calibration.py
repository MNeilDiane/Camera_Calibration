import cv2
import numpy as np
import glob
#找棋盘格角点
#阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#棋盘格模板规格
w = 9
h = 6
objp = np.zeros((w*h, 3), np.float32)
print(objp)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objpoints = []#现实世界的坐标点
imgpoints = []#二维图像的坐标点
images = glob.glob('calib/*.png')#your path
for fname in images:
    img = cv2.imread(fname)
    #灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #void cornerSubPix(InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria criteria)
        #image：输入图像
        # corners：输入角点的初始坐标以及精准化后的坐标用于输出。
        # winSize：搜索窗口边长的一半，例如如果winSize=Size(5,5)，则一个大小为的搜索窗口将被使用。
        # zeroZone：搜索区域中间的dead region边长的一半，有时用于避免自相关矩阵的奇异性。如果值设为(-1,-1)则表示没有这个区域。
        # criteria：角点精准化迭代过程的终止条件。也就是当迭代次数超过criteria.maxCount，或者角点位置变化小于criteria.epsilon时，停止迭代过程。
        objpoints.append(objp)
        imgpoints.append(corners)
        #显示角点
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imshow('findCorners', img)
        cv2.waitKey(1)
cv2.destroyAllWindows()

#标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#calibrateCamera(objectPoints, imagePoints,imageSize,
# cameraMatrix,distCoeffs,rvecs = None,tvecs = None,flags = None,criteria = None)
#objectPoints: array,世界坐标系中的点。imagePoints: array,其对应的图像点。
# imagesize:array,图像的大小，仅用于初始化相机的内参矩阵。
# cameraMatrix:array,输入/输出 3x3 的浮点相机内参矩阵 如果CALIB_USE_INTRINSIC_GUESS
# 或 CALIB_FIX_ASPECT_RATIO 被指定，在函数被调用之前，fx,fy,cx,cy中的部分或所有参数必须要被初始化。
#distCoeffs: array,输入或输出的系数数组。
#rvecs:array,旋转向量。
#tvecs:array,位移向量。
#flag:int,不同的值，是零或下列值的组合，参考官方文档
#criteria:迭代优化算法的终止准则。

#去畸变
img2 = cv2.imread('calib/1.png')#your path
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
#去畸变函数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

cv2.imwrite(('res.png', dst))
total_error = 0
for i in (len(objpoints)):
    imgpoints2,_ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #三维坐标点投影到二维坐标 objectPoints世界坐标系3D点的3维坐标
    #revc世界坐标系变换到相机坐标系的旋转向量，tvex...平移向量
    #mtx内参矩阵 dist畸变矩阵
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.MORM_l2)/len(imgpoints2)
    #L1范数是所有元素的绝对值的和；L2范数是所有元素(绝对值)的平方和再开方
    total_error += error
print("Total error:", total_error/len(objpoints))









import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay
import pandas as pd
import sys
import os

# 利用Dlib官方训练好的模型“shape_predictor_68_face_landmarks.dat”进行68点标定，
# 然后利用OpenCv进行图像化处理，在人脸上画出68个点
# 模型导入
# dlib关键点检测模型(68个)


# 68-point calibration using Dlib officially trained model "shape_predictor_68_face_landmarks.dat"
# Then use OpenCv for image processing and draw 68 points on the person's face.
#import model
# dlib key point detection model (68)

predictor_model = 'shape_predictor_68_face_landmarks.dat'

# Use dlib to get the feature points of the face
def get_points1(image):  # 用 dlib 来得到人脸的特征点

    # Same as face detection, using frontal_face_detector that comes with dlib as a face detector
    # 与人脸检测相同，使用dlib自带的frontal_face_detector作为人脸检测器
    face_detector = dlib.get_frontal_face_detector()  #for face detection, extracting the outer rectangle of the face
# 正向人脸检测器，进行人脸检测，提取人脸外部矩形框

    # 关键点提取需要一个特征提取器(predictor)，构建特征提取器可以训练模型
    # #使用官方提供的模型构建特征提取器

    # Keypoint extraction requires a feature extractor, and the feature extractor can be used to train the model.
    # #Build a feature extractor using the officially provided model
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print('No face detected in image {}'.format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)  # get landmark
    points = []

    # points为收集的脸部68个特征关键点坐标
    # points is the coordinates of the 68 key points of the collected face
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])
        p1 = pd.DataFrame(points)
        p1.to_csv('./image1.csv',index=False)

    a=1
    # 加入四个顶点和四条边的中点
    # Add four vertices and the midpoint of the four sides
    # 1:kweight 0 width
    x = image.shape[1] - 1
    y = image.shape[0] - 1
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])


    p1 = pd.DataFrame(points)
    p1.to_csv('./image1.csv',index=False)

    return np.array(points)

def get_points2(image):  # 用 dlib 来得到人脸的特征点

    # Same as face detection, using frontal_face_detector that comes with dlib as a face detector
    # 与人脸检测相同，使用dlib自带的frontal_face_detector作为人脸检测器
    face_detector = dlib.get_frontal_face_detector()  #for face detection, extracting the outer rectangle of the face
# 正向人脸检测器，进行人脸检测，提取人脸外部矩形框

    # 关键点提取需要一个特征提取器(predictor)，构建特征提取器可以训练模型
    # #使用官方提供的模型构建特征提取器

    # Keypoint extraction requires a feature extractor, and the feature extractor can be used to train the model.
    # #Build a feature extractor using the officially provided model
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print('No face detected in image {}'.format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)  # get landmark
    points = []

    # points为收集的脸部68个特征关键点坐标
    # points is the coordinates of the 68 key points of the collected face
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])
        #print(points)

    a=1
    # 加入四个顶点和四条边的中点
    # Add four vertices and the midpoint of the four sides

    x = image.shape[1] - 1
    y = image.shape[0] - 1
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])

    p2 = pd.DataFrame(points)
    p2.to_csv('./image2.csv',index=False)

    return np.array(points)

# Use Delaunay triangulation on feature points to join point sets into triangles of a certain size,
# and the allocation is relatively reasonable in order to present a beautiful triangulation
# 利用 get_points 方法 可以计算出一个新的集合C ，新的集合表示融合以后的图像中
# 这些点所在的坐标位置，新集合中的每个元素表示一个点。在这个新的集合C中运用
# Delaunay 三角剖分， Delaunay三角剖分的结果是新的集合中所有点构成的三角型
# 三角剖分中每个元素暗示了构成一个三角形的三个点在C中的位置索引，
def get_triangles(points):  #  在特征点上使用 Delaunay 三角剖分，将点集连接成一定大小的三角形，且分配要相对合理，才能呈现出漂亮的三角化
    p = Delaunay(points)
    return p.simplices


# img1_rect, tri_rect1, tri_rect_warped, size)

def affine_transform(input_image, input_triangle, output_triangle, size):  # affine transformation of the face to determine the position
    # 对人脸进行仿射变换，确定位置
    # This function uses three pairs of points to calculate the affine transformation, which is mainly used to generate the affine transformation matrix.
    # First parameter: the triangle vertex coordinates of the input image
    # second parameter: the corresponding triangle vertex coordinates of the output image

    # 计算仿射变换的目的：现在我们在image1中有76个点（集合A），image2中也有76个点（集合B）
    # 融合后的图像也有76个点，由这76个点所组成的三角形信息也存储在三角剖分的结果中。
    # 我们在image1中取出一个三角形，与image2中对应的三角形之间计算仿射变换（cv2.getAffineTransform）
    # 为image1 和image_goal 之间的每一对三角形计算仿射变换，最终也要在image2和image_goal之间也重复这步骤


    # 此函数 用 由三对点计算仿射变换 主要用于生成仿射变换矩阵
    # 第一个参数：输入图像的三角形顶点坐标
    # 第二个参数：输出图像的相应的三角形顶点坐标


    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle))

    # 对图像做仿射变换
    #第一个参数：输入图像 第二个参数：输出图像
    # 第三个参数：2×3 变换矩阵
    # 输出图像的大小
    # flags=双线性插值（默认方法）
    # 边界处理方式

    # affine transformation of images
    # First: Input image Second parameter: Output image
    # Third parameter: 2×3 transformation matrix
    # Output image size
    # flags=Bilinear interpolation (default method)
    # boundary processing method

    # 使用 warpAffine 的目的： 对于image1 中的每一个三角形，
    # 利用 getAffineTransform() 可以得到一个仿射变换矩阵，
    # 将三角形中的所有像素扭曲到目标图像中的对应三角形中，
    # 重复所有三角形得到一个 image1 的扭曲版本。
    # 同样的操作得到一个image2 的扭曲版本
    # 但是之后需要对图像进行处理 165行代码

    # 在利用函数warpAffine
    # 的时候要使用荣火热模式
    # BORDER_REFLECT_101，
    # 该模式可以很好的隐藏缝隙

    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image


def morph_triangle(img1, img2, img, tri1, tri2, tri, alpha):  # Triangle deformation, Alpha blend
# 三角形变形，Alpha 混合
    # 计算三角形的边界框  计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的

# Calculate the bounding box of the triangle Calculate the minimum rectangle of the vertical boundary of the outline,
# which is parallel to the upper and lower boundaries of the image
    # Find the coordinates of the upper left corner of tri1,
#       and the length and width of tri1
    # 寻找tri1的左上角坐标，和tri1的长和宽
    # 矩形边框（Bounding Rectangle）
    # 用一个最小的矩形，把找到的形状包起来。其中还有一个带旋转的矩形，但是面积比外边包围矩阵更小
    rect1 = cv2.boundingRect(np.float32([tri1]))
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []

    for i in range(0, 3):
        tri_rect_warped.append(
            ((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(
            ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(
            ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

    # 在边界框内进行仿射变换 Affine transformation in the bounding box
    img1_rect = img1[rect1[1]:rect1[1] +
                     rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] +
                     rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
#img1_rect is input_image and tri_rect1 is input_triangle
# and tri_rect_warped is output_triangle
    warped_img1 = affine_transform(
        img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(
        img2_rect, tri_rect2, tri_rect_warped, size)

    # 加权求和
    # 有了关键点，相当于我们有了两张脸的数据，
    # 接下来我们将针对于这些关键点进行融合，
    # 融合的公式代码如下所示：

    #weighted summation
    #Having a key point is equivalent to having two faces of data.
    # Next we will focus on these key points,
    # The formula code for the fusion is as follows:

    # points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

    # 因为warpAffine是对一幅图像进行操作而不是三角形，
    # 一个可行的办法是为每个三角形求一个最小包围矩形，
    # 对包围矩形利用warpAffine进行扭曲，
    # 然后利用掩模操作将三角形以外的区域去掉。
    # 三角形掩模可以利用fillConvexPoly
    # （函数FillConverExpoly根据多边形顶点绘制一个填充的凸多边形）创建。

    # 生成模板 Generate template
    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    # 函数fillConvexPoly填充凸多边形内部
    # 第一个参数：图像模板 第二个参数：指向单个多边形的指针数组
    # 第三个参数：凸多边形的顶点。

    #function fillConvexPoly fills the interior of the convex polygon
    # first parameter: image template second parameter: pointer array to a single polygon
    # Third parameter: the vertex of the convex polygon.
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

    # Application template
    img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
            rect[2]] * (1 - mask) + img_rect * mask



def morph_faces(filename1, filename2, alpha=0.5):  # Fusion picture

    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    # scr:original image  dsize：the size of output image(img1.shape[1],img1.shape[0]) interpolation：the method of interpolation
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]),interpolation=cv2.INTER_CUBIC)
    print('img1.shape',img1.shape)
    print('img2.shape',img2.shape)

    #通过自定义的 get_points 方法得到 人脸特征点  包括 pointes收集的脸部68个特征关键点坐标
    # Get the face feature points by the custom get_points method,
    # including the 68 feature key coordinates of the face collected by pointes

    points1 = get_points1(img1)

    # Read in the points from a text file

    #print('pionts1:',len(points1),points1)
    points2 = get_points2(img2)


    # with open("image2.txt") as file:
    #     for line in file:
    #         x, y = line.split()
    #         points2.append((int(x), int(y)))
    #
    #print('pionts2:', len(points2), points2)


    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)

    p = pd.DataFrame(points)

    with open("image_morph.txt") as file:
        for line in file:
            x, y = line.split()
            p.append((int(x), int(y)))



    p.to_csv('./1.csv',index=False)

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)


    # 在特征点上使用 Delaunay 三角剖分，将点集连接成一定大小的三角形，且分配要相对合理，才能呈现出漂亮的三角化
    # Use Delaunay triangulation on feature points to join point sets into triangles of a certain size,
    # and the allocation is relatively reasonable in order to present a beautiful triangulation

    triangles = get_triangles(points)
    for i in triangles:
        x = i[0]
        y = i[1]
        z = i[2]

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]
        morph_triangle(img1, img2, img_morphed, tri1, tri2, tri, alpha)

    return np.uint8(img_morphed)



def main(file1,file2,alpha):
    try:
     alpha = float(alpha)
    except:
        alpha = 0.5
    img_morphed = morph_faces(file1, file2, alpha)
    output_file = '{}_{}_{}.jpg'.format(
        file1.split('.')[0][-2:], file2.split('.')[0][-1:], alpha)
    cv2.imwrite(output_file, img_morphed)
    return output_file



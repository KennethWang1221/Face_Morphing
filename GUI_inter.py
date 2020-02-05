from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import *
import PIL
import cv2
from face_morhper import *


root = Tk()
root.title('Image Morphing')


def center_window(w, h):
    # 获取屏幕 宽、高
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    # 计算 x, y 位置
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
center_window(1500, 1000)


# 第4步，在图形界面上设定标签
var = StringVar()  # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
l = Label(root, textvariable=var, bg='white', fg='blue', font=('Arial', 20), width=100, height=100)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l.pack()


#
# # set whitebackground.jpg as GUI background
# decoration = PIL.Image.open('/Users/wangxiang/Code/Pycharm_Project/demo1/12.jpg').resize((1200, 600))
# render = ImageTk.PhotoImage(decoration)
# img = Label(image=render)
# img.image = render
# img.place(x=0, y=0)

global path1_, path2_, rate, seg_img_path


# show Image1
def show_original1_pic():
    global path1_
    path1_ = askopenfilename(title='Upload Image')
    print(path1_)
    Img = PIL.Image.open(r'{}'.format(path1_))
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.image = img_png_original  # keep a reference
    cv_orinial1.create_image(5, 5, anchor='nw',
                             image=img_png_original)  # anchor='nw',so img_png_original anchor for bottom left


# 原图2展示
def show_original2_pic():
    global path2_
    path2_ = askopenfilename(title='Choose Image')
    print(path2_)
    Img = PIL.Image.open(r'{}'.format(path2_))
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original2.image = img_png_original  # keep a reference
    cv_orinial2.create_image(5, 5, anchor='nw', image=img_png_original)


# image_file = tk.PhotoImage(file='1.gif')`这一句是创造一个变量存放`1.gif`这张图片。
# `image = canvas.create_image(10, 10, anchor='nw', image=image_file)`里面的参数`10,10`就是图片放入画布的坐标，
# 而这里的`anchor=nw`则是把图片的左上角作为锚定点，在加上刚刚给的坐标位置，即可将图片位置确定。
# 最后一个参数的意思大家应该都知道，就是将刚刚存入的图片变量，赋值给`image`。

# 人脸融合效果展示
def show_morpher_pic():
    global path1_, seg_img_path, path2_

    # Python Tkinter 文本框用来让用户输入一行文本字符串。
    # print(entry.get())
    mor_img_path = main(path1_, path2_, 0.5)
    # mor_img_path = main(path1_,path2_,entry.get())
    Img = PIL.Image.open(r'{}'.format(mor_img_path))
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_seg = ImageTk.PhotoImage(Img)
    label_Img_seg.config(image=img_png_seg)
    label_Img_seg.image = img_png_seg  # keep a reference


def show_points1():


    Img = PIL.Image.open('/Users/wangxiang/Code/Pycharm_Project/demo1/Trudeau_points.jpg')
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.image = img_png_original  # keep a reference
    cv_orinial1.create_image(5, 5, anchor='nw',
                             image=img_png_original)  # anchor='nw',so img_png_original anchor for bottom left


# def show_points1():
#     img = cv2.imread("/Users/wangxiang/Code/Pycharm_Project/demo1/Trudeau_points.jpg")
#
#     while True:
#
#         cv2.imshow("Trudeau_points.jpg", img)
#
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break;
#
#     #cv2.destroyAllWindows()
#
#     #
#     # img_gif = ImageTk.PhotoImage(file='Trudeau_points.jpg')
#     # label_img = ImageTk.Label(root, image=img_gif)
#     # label_img.pack()


def show_points2():
    Img = PIL.Image.open('/Users/wangxiang/Code/Pycharm_Project/demo1/tom_points.jpg')
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.image = img_png_original  # keep a reference
    cv_orinial1.create_image(5, 5, anchor='nw',
                             image=img_png_original)  # anchor='nw',so img_png_original anchor for bottom left




def show_Delauny1():
    Img = PIL.Image.open('/Users/wangxiang/Code/Pycharm_Project/demo1/Del_Img1.jpg')
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.image = img_png_original  # keep a reference
    cv_orinial1.create_image(5, 5, anchor='nw',
                             image=img_png_original)  # anchor='nw',so img_png_original anchor for bottom left

    # img = cv2.imread("/Users/wangxiang/Code/Pycharm_Project/demo1/Del_Img1.jpg")
    #
    # cv2.imshow("Del_Img1.jpg", img)
    #
    # cv2.waitKey(0)

def show_Delauny2():
    Img = PIL.Image.open('/Users/wangxiang/Code/Pycharm_Project/demo1/tom_Del.jpg')
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 调整图片大小至256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.image = img_png_original  # keep a reference
    cv_orinial1.create_image(5, 5, anchor='nw',
                             image=img_png_original)  # anchor='nw',so img_png_original anchor for bottom left

    # img = cv2.imread("/Users/wangxiang/Code/Pycharm_Project/demo1/tom_Del.jpg")
    #
    # cv2.imshow("tom_Del.jpg", img)
    #
    # cv2.waitKey(0)


def quit():
    root.destroy()


# 原图1的展示
Button(root, bg='blue', text="Open Image1", command=show_original1_pic).place(x=260, y=440)
# 原图2的展示
Button(root, bg='blue', text="Open Image2", command=show_original2_pic).place(x=600, y=440)
# 进行提取结果的展示
Button(root, bg='red', text="Face Morphing", command=show_morpher_pic).place(x=920, y=440)

Button(root, bg='blue', text="Show_Points1", command=show_points1).place(x=260, y=500)

Button(root, bg='blue', text="Show_Points2", command=show_points2).place(x=600, y=500)

Button(root, bg='blue', text="Show_Delaunay1", command=show_Delauny1).place(x=260, y=560)

Button(root, bg='blue', text="Show_Delaunay2", command=show_Delauny2).place(x=600, y=560)

Button(root, bg='blue', text="Exit", command=quit).place(x=1100, y=440)


# Label(root,text = "alpha coefficient",font=10).place(x=50,y=10)
# entry = Entry(root)
# entry.place(x=130,y=10)

Label(root, text="Image Morphing --Wang Xiang").place(x=550, y=60)

Label(root, text="Image1", font=10).place(x=280, y=120)
cv_orinial1 = Canvas(root, bg='white', width=270, height=270)
cv_orinial1.place(x=180, y=150)
label_Img_original1 = Label(root)
label_Img_original1.place(x=180, y=150)

Label(root, text="Image2", font=10).place(x=600, y=120)
cv_orinial2 = Canvas(root, bg='white', width=270, height=270)
cv_orinial2.place(x=500, y=150)
label_Img_original2 = Label(root)
label_Img_original2.place(x=500, y=150)

Label(root, text="Image_Morphing", font=10).place(x=920, y=120)
cv_seg = Canvas(root, bg='white', width=270, height=270)
# cv_seg.create_rectangle(8,8,260,260,width=1,outline='blue')
cv_seg.place(x=820, y=150)
label_Img_seg = Label(root)
label_Img_seg.place(x=820, y=150)

root.mainloop()
# -*- coding:utf-8 -*-



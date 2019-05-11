# coding: utf-8
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

def football_level():
    # 输入数据
    data = pd.read_csv('./data/football.csv', encoding='gbk')
    train_x = data[["2019年国际排名","2018世界杯","2015亚洲杯"]]
    df = pd.DataFrame(train_x)
    kmeans = KMeans(n_clusters=3)
    # 规范化到 [0,1] 空间
    min_max_scaler=preprocessing.MinMaxScaler()
    train_x=min_max_scaler.fit_transform(train_x)
    # kmeans 算法
    kmeans.fit(train_x)
    predict_y = kmeans.predict(train_x)
    # 合并聚类结果，插入到原数据中
    result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
    result.rename({0:u'聚类'},axis=1,inplace=True)
    print(result)


def image_grey_segment():
    import PIL.Image as image
    # 加载图像，并对数据进行规范化
    def load_data(filePath):
        # 读文件
        f = open(filePath, 'rb')
        data = []
        # 得到图像的像素值
        img = image.open(f)
        # 得到图像尺寸
        width, height = img.size
        for x in range(width):
            for y in range(height):
                # 得到点 (x,y) 的三个通道值
                c1, c2, c3 = img.getpixel((x, y))
                data.append([c1, c2, c3])
        f.close()
        # 采用 Min-Max 规范化
        mm = preprocessing.MinMaxScaler()
        data = mm.fit_transform(data)
        return np.mat(data), width, height

    # 加载图像，得到规范化的结果 img，以及图像尺寸
    img, width, height = load_data('./data/weixin.jpg')
    # 用 K-Means 对图像进行 2 聚类
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(img)
    label = kmeans.predict(img)
    # 将图像聚类结果，转化成图像尺寸的矩阵
    label = label.reshape([width, height])
    # 创建个新图像 pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
    pic_mark = image.new("L", (width, height))
    for x in range(width):
        for y in range(height):
            # 根据类别设置图像灰度, 类别 0 灰度值为 255， 类别 1 灰度值为 127
            pic_mark.putpixel((x, y), int(256 / (label[x][y] + 1)) - 1)
    pic_mark.save("./data/weixin_mark.jpg", "JPEG")

    print(image)

def image_color_segment():
    import PIL.Image as image
    from skimage import color

    def load_data(filePath):
        f = open(filePath, 'rb')
        data = []
        img = image.open(f)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                c1, c2, c3 = img.getpixel((x, y))
                data.append([c1, c2, c3])
        f.close()
        mm = preprocessing.MinMaxScaler()
        data = mm.fit_transform(data)
        return np.mat(data), width, height

    img, width, height = load_data('./data/weixin.jpg')
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(img)
    label = kmeans.predict(img)
    label = label.reshape([width, height])
    label_color = (color.label2rgb(label) * 255).astype(np.uint8)
    label_color = label_color.transpose(1,0,2)
    images = image.fromarray(label_color)

    print(label_color.shape)

    images.save("./data/weixin_mark_color.jpg", "JPEG")
    print(image)


def image_segment_back():
    import PIL.Image as image
    from skimage import color
    def load_data(filePath):
        f = open(filePath, 'rb')
        data = []
        img = image.open(f)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                c1, c2, c3 = img.getpixel((x, y))
                data.append([(c1 + 1) / 256.0, (c2 + 1) / 256.0, (c3 + 1) / 256.0])
        f.close()
        return np.mat(data), width, height

    # 加载图像，得到规范化的结果imgData，以及图像尺寸
    img, width, height = load_data('./data/weixin.jpg')
    kmeans = KMeans(n_clusters=16)
    label = kmeans.fit_predict(img)
    label = label.reshape([width, height])
    img = image.new('RGB', (width, height))
    for x in range(width):
        for y in range(height):
            c1 = kmeans.cluster_centers_[label[x, y], 0]
            c2 = kmeans.cluster_centers_[label[x, y], 1]
            c3 = kmeans.cluster_centers_[label[x, y], 2]
            img.putpixel((x, y), (int(c1 * 256) - 1, int(c2 * 256) - 1, int(c3 * 256) - 1))
    img.save('./data/weixin_back.jpg')

image_segment_back()

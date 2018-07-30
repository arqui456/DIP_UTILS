import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def readGrayScaleImg(input):
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", img)

def loadColor(input):
    img = cv2.imread(str(input), cv2.IMREAD_COLOR)
    cv2.imshow("img", img)

def loadColorWithChannels(input):
    img = cv2.imread(str(input), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.split(img)
    plt.subplot("141"); plt.title("Original"); plt.imshow(img)
    plt.subplot("142"); plt.title("R");  plt.imshow(rgb[0], 'gray')
    plt.subplot("143"); plt.title("G");  plt.imshow(rgb[1], 'gray')
    plt.subplot("144"); plt.title("B");  plt.imshow(rgb[2], 'gray')
    plt.show()

def histogramGray(input):
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    plt.subplot("211"); plt.title("Original"); plt.imshow(img)
    plt.subplot("212"); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 255])
    plt.show()

def histogramColor(input):
    img = cv2.imread(str(input), cv2.IMREAD_COLOR)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def createImg(input):
    if input == 0:
        """ create a img filled with zeros"""
        img = np.zeros((500, 500), dtype=np.int16)
    elif input == 1:
        """ create a img filled with ones"""
        img = np.ones((500, 500), dtype=np.float32)
    elif input == 2:
        """ create a img filled with a scalar"""
        img = 127 * np.ones((500, 500), dtype=np.int16)
    elif input == 3:
        """ Initializing a grayscale image with random values, uniformly distributed"""
        img = np.ones((250, 250), dtype=np.uint8)
        cv2.randu(img, 0, 255)
    elif input == 4:
        """ Initializing a color image with random values, uniformly distributed """
        img = np.ones((250, 250, 3), dtype=np.uint8)
        bgr = cv2.split(img)
        cv2.randu(bgr[0], 0, 255)
        cv2.randu(bgr[1], 0, 255)
        cv2.randu(bgr[2], 0, 255)
        img = cv2.merge(bgr)
    elif input == 5:
        """ Initializing a grayscale image with random values, normally distributed """
        img = np.ones((250, 250), dtype=np.uint8)
        cv2.randn(img, 127, 40)
    elif input == 6:
        """ Initializing a color image with random values, normally distributed """
        img = np.ones((250, 250, 3), dtype=np.uint8)
        bgr = cv2.split(img)    
        cv2.randn(bgr[0], 127, 40)
        cv2.randn(bgr[1], 127, 40)
        cv2.randn(bgr[2], 127, 40)
        img = cv2.merge(bgr)
    elif input == 7:
        """ Initialize a color grayscale with uniformly distributed random values and visualize its histogram """
        img = np.ones((250, 250), dtype=np.uint8)
        cv2.randu(img, 0, 255)
        plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
        plt.show()
        return
    elif input == 8:
        """ Initialize a color image with uniformly distributed random values and visualize its histogram """
        img = np.ones((250, 250, 3), dtype=np.uint8)
        bgr = cv2.split(img)
        cv2.randu(bgr[0], 0, 255)
        cv2.randu(bgr[1], 0, 255)
        cv2.randu(bgr[2], 0, 255)
        img = cv2.merge(bgr)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()
        return
    elif input == 9:
        """ Initialize a grayscale image with normally distributed random values and visualize its histogram """
        img = np.ones((250, 250), dtype=np.uint8)
        cv2.randn(img, 127, 40)
        plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
        plt.show()
        return
    elif input == 10:
        """ Initialize a color image with normally distributed random values and visualize its histogram """
        img = np.ones((250, 250, 3), dtype=np.uint8)
        bgr = cv2.split(img)
        cv2.randn(bgr[0], 127, 40)
        cv2.randn(bgr[1], 127, 40)
        cv2.randn(bgr[2], 127, 40)
        img = cv2.merge(bgr)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()
        return
    elif input == 11:
            """  Convert image to different ranges """
            img = np.ones((3, 3), dtype=np.float32)
            cv2.randn(img, 0, 1)
            print("Normally distributed random values = \n", img, "\n\n")
            cv2.normalize(img, img, 255, 0, cv2.NORM_MINMAX)
            print("Normalized = \n", img, "\n\n")
            img = np.asarray(img, dtype=np.uint8)
            print("Converted to uint8 = \n", img, "\n\n")
            img = 255 * img
            img = np.asarray(img, dtype=np.uint8)
            print(img, "\n\n")
            return
    elif input == 12:
            """ Create random images continuously """
            img = np.ones((250, 250), dtype=np.uint8)
            while cv2.waitKey(0) != ord('q'):
                cv2.randn(img, 120, 60)
                cv2.imshow("img", img)
            return
    cv2.imshow("img", img)

def addScalar(input):
    """ add a scalar to an image """
    img = cv2.imread("img/lena.png", cv2.IMREAD_COLOR)
    val = 100
    img2 = img + val
    print(img)
    print(img2)
    plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

def mergeImages(input):
    """ add two images """
    img_temp1, img_temp2 = input
    img = cv2.imread(str(img_temp1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img_temp2), cv2.IMREAD_COLOR)
    img3 = img + img2
    plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

def imageMax(input):
    """ takes two images and make a third one appliying max to each pixel """
    img_temp1, img_temp2 = input
    img = cv2.imread(str(img_temp1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img_temp2), cv2.IMREAD_COLOR)
    img3 = cv2.max(img, img2)
    plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

def imageAbs(input):
    """ absolut diff between two imgs"""
    img_temp1, img_temp2 = input
    img = cv2.imread(str(img_temp1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img_temp2), cv2.IMREAD_COLOR)
    img3 = cv2.absdiff(img, img2)
    plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

def  videoDiff():
    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) != ord('q'):
        _, frame1 = cap.read()
        _, frame2 = cap.read()

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        cv2.imshow('Gray 1', gray1)
        cv2.imshow('Gray 2', gray2)
        cv2.imshow('DIFF', diff)

    cap.release()
    cv2.destroyAllWindows()

def addNoise(input):
    img = cv2.imread(str(input), cv2.IMREAD_COLOR)
    noise = np.zeros(img.shape, img.dtype)
    cv2.randn(noise, 0, 150)
    img2 = img + noise
    plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

def addSaltAndPepper(input):
    img = cv2.imread(str(input), cv2.IMREAD_COLOR)
    noise = np.zeros((img.shape[0], img.shape[1]), img.dtype)
    cv2.randu(noise, 0, 255)
    salt = noise > 250
    pepper = noise < 5
    img2 = img.copy()
    img2[salt == True] = 255
    img2[pepper == True] = 0
    plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

def setOP(input):
    """ TODO, switch between OR , AND and NOT"""
    img_temp1, img_temp2 = input
    img = cv2.imread(str(img_temp1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img_temp2), cv2.IMREAD_GRAYSCALE)
    and_img = img & img2
    or_img = img | img2
    not_img = ~img
    plt.subplot("151"); plt.title("IMG 1"); plt.imshow(img, 'gray')
    plt.subplot("152"); plt.title("IMG 2"); plt.imshow(img2, 'gray')
    plt.subplot("153"); plt.title("AND"); plt.imshow(and_img, 'gray')
    plt.subplot("154"); plt.title("OR"); plt.imshow(or_img, 'gray')
    plt.subplot("155"); plt.title("NOT"); plt.imshow(not_img, 'gray')
    plt.show()

def imgNegative(input):
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", 255 - img)

def logTransform(input):
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    img2 = np.ones(img.shape, np.float64)
    c = 1
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            intensity = img[x][y]
            intensity_new = c * np.log(1 + intensity)
            img2[x][y] = intensity_new
    cv2.normalize(img2, img2, 255, 0, cv2.NORM_MINMAX)
    plt.subplot("221"); plt.title("Image 1"); plt.imshow(img, "gray")
    plt.subplot("222"); plt.title("Hist 1"); plt.hist(img.ravel(), 256, [0, 255])
    plt.subplot("223"); plt.title("Image 2"); plt.imshow(img2, "gray")
    plt.subplot("224"); plt.title("Hist 2"); plt.hist(img2.ravel(), 256, [0, 255])
    plt.show()

def intensityTransform(input):
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    img2 = np.ones(img.shape, np.uint8)
    cv2.namedWindow("img")
    cv2.namedWindow("img2")
    n = 0
    cv2.createTrackbar("n", "img2", n, 10, doNothing)
    while cv2.waitKey(1) != ord('q'):
        n = cv2.getTrackbarPos("n", "img2")
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                intensity = img[x][y]
                intensity_new = np.power(intensity, n)
                img2[x][y] = intensity_new
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)

def piceWiseTransform(input):
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256), 0, 0, cv2.INTER_LINEAR)
    img2 = np.copy(img)
    hist = np.copy(img)
    hist2 = np.copy(img)
    T0 = 255 * np.ones(img.shape, np.uint8)
    cv2.namedWindow("Transformation", cv2.WINDOW_AUTOSIZE)
    r1 = 65
    s1 = 65
    r2 = 195
    s2 = 195
    cv2.createTrackbar("r1", "Transformation", r1, T0.shape[0] - 1, doNothing)
    cv2.createTrackbar("s1", "Transformation", s1, T0.shape[0] - 1, doNothing)
    cv2.createTrackbar("r2", "Transformation", r2, T0.shape[0] - 1, doNothing)
    cv2.createTrackbar("s2", "Transformation", s2, T0.shape[0] - 1, doNothing)
    while True:
        r1 = cv2.getTrackbarPos("r1", "Transformation")
        s1 = cv2.getTrackbarPos("s1", "Transformation")
        r2 = cv2.getTrackbarPos("r2", "Transformation")
        s2 = cv2.getTrackbarPos("s2", "Transformation")
        T = np.copy(T0)
        p1 = (r1, T.shape[1] - 1 - s1)
        p2 = (r2, T.shape[1] - 1 - s2)
        cv2.line(T, (0, T.shape[0] - 1), p1, (0, 0, 0), 2, cv2.LINE_8, 0)
        cv2.circle(T, p1, 4, 0, 2, cv2.LINE_8, 0)
        cv2.line(T, p1, p2, 0, 2, cv2.LINE_8, 0)
        cv2.circle(T, p2, 4, 0, 2, cv2.LINE_8, 0)
        cv2.line(T, p2, (T.shape[0] - 1, 0), 0, 2, cv2.LINE_8, 0)
        r = 0
        s = 0
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                r = img[y][x]
                s = compute_piecewise_linear_val(r / 255.0, r1 / 255.0, s1 / 255.0, r2 / 255.0, s2 / 255.0)
                img2[y][x] = 255.0 * s
        hist = compute_histogram_1C(img)
        hist2 = compute_histogram_1C(img2)
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        cv2.imshow("hist", hist)
        cv2.imshow("hist2", hist2)
        cv2.imshow("Transformation", T)
        if cv2.waitKey(1) == ord('q'):
            break
def thresh1(input):
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    img2 = np.ones(img.shape, img.dtype)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img2[x][y] = img[x][y] if (150 < img[x][y] < 240) else 0
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)

def thresh2(input):
    cv2.namedWindow("img")
    cv2.namedWindow("img2")
    cv2.namedWindow("result")
    lower = 0
    upper = 255
    cv2.createTrackbar("lower", "result", lower, 255, doNothing)
    cv2.createTrackbar("upper", "result", upper, 255, doNothing)

    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("img/kidney_little.tif", cv2.IMREAD_GRAYSCALE)
    img2 = np.copy(img)

    last_lower = 0
    last_upper = 0

    while cv2.waitKey(1) != ord('q'):

        lower = cv2.getTrackbarPos("lower", "result")
        upper = cv2.getTrackbarPos("upper", "result")

        if last_lower != lower or last_upper != upper:
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    img2[x][y] = img[x][y] if (lower < img[x][y] < upper) else 0

            # _, img2 = cv2.threshold(img, lower, upper, cv2.THRESH_BINARY)
            # img2 = cv2.bitwise_and(result, img)

        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        cv2.imshow("result", img2)

        last_lower = lower
        last_upper = upper

def staticThresh(input):
    cv2.namedWindow("img2")

    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    img2 = np.copy(img)

    threshType = 0
    thresh = 127

    cv2.createTrackbar("threshType", "img2", threshType, 4, doNothing)
    cv2.createTrackbar("thresh", "img2", thresh, 255, doNothing)

    # 0 - THRESH_BINARY
    # 1 - THRESH_BINARY_INV
    # 2 - THRESH_TRUNC
    # 3 - THRESH_TOZERO
    # 4 - THRESH_TOZERO_INV

    while True:
        threshType = cv2.getTrackbarPos("threshType", "img2")
        thresh = cv2.getTrackbarPos("thresh", "img2")

        _, img2 = cv2.threshold(img, thresh, 255, threshType)
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)

        if cv2.waitKey(1) == ord('q'):
            break
def bitSlicing(input):
    cv2.namedWindow("img")
    cv2.namedWindow("img2")
    
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    img2 = np.copy(img)
    
    slice = 7
    cv2.createTrackbar("slice", "img2", slice, 7, doNothing)
    
    while True:
    
        slice = cv2.getTrackbarPos("slice", "img2")
    
        # cv2.bitwise_and(img, 0b00000001, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0b00000010, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0b00000100, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0b00001000, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0b00010000, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0b00100000, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0b01000000, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0b10000000, img2) # Using only the four more significant bits.
    
        img2 = cv2.bitwise_and(img, 2 << slice, img2)
        # cv2.bitwise_and(img, 0xf0, img2) # Using only the four more significant bits.
        # cv2.bitwise_and(img, 0xd0, img2) # Using only the three more significant bits.
        # cv2.bitwise_and(img, 0xc0, img2) # Using only the two more significant bits.
        # cv2.bitwise_and(img, 0x80, img2) # Using only the most significant bit.
    
        img2 = np.asarray(img2, np.float32)
        cv2.normalize(img2, img2, 0, 1, cv2.NORM_MINMAX)
        # img2 = 255 * img2
        # img2.convertTo(img2, CV_8U)
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        if cv2.waitKey(1) == ord('q'):
            break

def histEq(input):
    #Histogram equalization
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.equalizeHist(img)
    hist = compute_histogram_1C(img)
    hist2 = compute_histogram_1C(img2)
    plt.subplot("221"); plt.title("Image 1"); plt.imshow(img, "gray")
    plt.subplot("222"); plt.title("Hist 1"); plt.imshow(hist, "gray")
    plt.subplot("223"); plt.title("Image 2"); plt.imshow(img2, "gray")
    plt.subplot("224"); plt.title("Hist 2"); plt.imshow(hist2, "gray")
    plt.show()

def LHP(input):
    # Local Histogram Processing
    img = cv2.imread("img/squares_noisy.tif", cv2.IMREAD_GRAYSCALE)
    img2 = np.zeros(img.shape, img.dtype)
    wsize = 1

    for x in range(wsize, img.shape[0] - wsize):
        for y in range(wsize, img.shape[1] - wsize):
            cv2.equalizeHist(img[y - wsize: y + wsize][x - wsize: x + wsize],
                             img2[y - wsize: y + wsize][x - wsize: x + wsize])

    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    while cv2.waitKey(1) != ord('q'):
        pass

def LHP1(input):
    cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("New", cv2.WINDOW_KEEPRATIO)
    img = cv2.imread('img/tungsten.tif', cv2.IMREAD_GRAYSCALE)

    img2 = np.copy(img)

    avg_global, std_global = cv2.meanStdDev(img)

    avg_global = avg_global[0][0]
    std_global = std_global[0][0]

    E = 3.0
    k0 = 0.4
    k1 = 0.02
    k2 = 0.4
    wsize = 1

    for x in range(wsize, img.shape[0] - wsize):
        for y in range(wsize, img.shape[1] - wsize):
            avg_local, std_local = cv2.meanStdDev(img[x-wsize:x+wsize, y-wsize:y+wsize])

            avg_local = avg_local[0][0]
            std_local = std_local[0][0]

            intensity = img[x, y]
            intensity_new = E*intensity if ((avg_local <= k0*avg_global) and (k1*std_global <= std_local) and (std_local <= k2*std_global)) else intensity
            img2[x, y] = intensity_new

    cv2.imshow("Original", img)
    cv2.imshow("New", img2)

    while cv2.waitKey(1) != ord('q'):
        pass

def FDO(input):

    #First derivative operators - Sobel masks - Part I

    cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("New", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Gx", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Gy", cv2.WINDOW_KEEPRATIO)
    
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    
    img2 = np.float64(img)
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img2[x,y] = np.float64(img[x,y])
    
    kx = [[-1,  0,  1],
          [-2,  0,  2],
          [-1,  0,  1]]
    kx = np.array(kx)
    
    ky = [[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]
    ky = np.array(ky)
    
    gx = cv2.filter2D(img2, -1, kx, cv2.BORDER_DEFAULT)
    gy = cv2.filter2D(img2, -1, ky, cv2.BORDER_DEFAULT)
    g = np.abs(gx) + np.abs(gy)
    
    cv2.normalize(gx, gx, 1, 0, cv2.NORM_MINMAX)
    cv2.normalize(gy, gy, 1, 0, cv2.NORM_MINMAX)
    cv2.normalize(g, g, 1, 0, cv2.NORM_MINMAX)
    
    
    while cv2.waitKey(1) != ord('q'):
        cv2.imshow("Original", img)
        cv2.imshow("New", g)
        cv2.imshow("Gx", gx)
        cv2.imshow("Gy", gy)

def FDO2(input):
    #First derivative operators - Sobel masks - Part II

    img = cv2.imread("img/lena.png", cv2.IMREAD_GRAYSCALE)
    
    gx, gy = cv2.spatialGradient(img, ksize=3, borderType=cv2.BORDER_DEFAULT)
    g = np.abs(gx) + np.abs(gy)
    
    gx = scaleImage2_uchar(gx)
    gy = scaleImage2_uchar(gy)
    g = scaleImage2_uchar(g)
    
    while cv2.waitKey(1) != ord('q'):
        cv2.imshow("Original", img)
        cv2.imshow("New", g)
        cv2.imshow("Gx", gx)
        cv2.imshow("Gy", gy)

def FDO3(input):
    #First derivative operators - Sobel masks - Part III

    img = cv2.imread("img/lena.png", cv2.IMREAD_GRAYSCALE)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    g = np.abs(gx) + np.abs(gy)

    gx = scaleImage2_uchar(gx)
    gy = scaleImage2_uchar(gy)
    g = scaleImage2_uchar(g)

    while cv2.waitKey(1) != ord('q'):
        cv2.imshow("Original", img)
        cv2.imshow("New", g)
        cv2.imshow("Gx", gx)
        cv2.imshow("Gy", gy)

def LaplacOP(input):
    #Image sharpening using the Laplacian operator - Part I

    cv2.namedWindow("img3")

    img = cv2.imread("img/lena.png", cv2.IMREAD_GRAYSCALE)

    img = np.float32(img)

    kernel = [[1.0,  1.0, 1.0],
              [1.0, -8.0, 1.0],
              [1.0,  1.0, 1.0]]
    kernel = np.array(kernel)

    img2 = cv2.filter2D(img, -1, kernel, cv2.BORDER_DEFAULT)
    img3 = np.copy(img2)

    cv2.normalize(img2, img2, 1, 0, cv2.NORM_MINMAX)

    factor = 5
    cv2.createTrackbar("factor", "img3", factor, 1000, doNothing)

    while cv2.waitKey(1) != ord('q'):
        factor = cv2.getTrackbarPos("factor", "img3")

        img3 = img - factor * img2
        hist = compute_histogram_1C(img3)

        cv2.imshow("img", scaleImage2_uchar(img))
        cv2.imshow("img2", scaleImage2_uchar(img2))
        cv2.imshow("img3", scaleImage2_uchar(img3))
        cv2.imshow("hist", hist)

def LaplacOP2(input):
    #Image sharpening using the Laplacian operator - Part II
    img = cv2.imread("img/lena.png", cv2.IMREAD_GRAYSCALE)
    lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=1, scale=1, delta=0)

    img = np.float32(img)
    lap = np.float32(lap)

    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)

    img2 = img - lap

    while cv2.waitKey(1) != ord('q'):
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        cv2.imshow("lap", scaleImage2_uchar(lap))



def main():
    print("Welcome to PDI tools 0.1!")
    #piceWiseTransform("lena.png")
    LaplacOP2("lena.png")

if __name__ == "__main__":
    main()
    #cv2.namedWindow("img")

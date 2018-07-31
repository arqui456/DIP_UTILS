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
    img = cv2.imread("eye.jpeg", cv2.IMREAD_GRAYSCALE)

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

    img = cv2.imread("eye.jpeg", cv2.IMREAD_GRAYSCALE)
    
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

    img = cv2.imread("eye.jpeg", cv2.IMREAD_GRAYSCALE)

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

    img = cv2.imread("eye.jpeg", cv2.IMREAD_COLOR)

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
    img = cv2.imread("eye.jpeg", cv2.IMREAD_COLOR)
    
    #IMREAD_GRAYSCALE
    lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=1, scale=1, delta=0)

    img = np.float32(img)
    lap = np.float32(lap)

    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)

    img2 = img - lap

    while cv2.waitKey(1) != ord('q'):
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        cv2.imshow("lap", scaleImage2_uchar(lap))

def gaussFilter(input):
    rows = 400
    cols = 400
    theta = 0
    xc = 200
    yc = 200
    sx = 120
    sy = 40

    cv2.namedWindow('img')
    cv2.createTrackbar("xc", "img", xc, int(rows), doNothing)
    cv2.createTrackbar("yc", "img", yc, int(cols), doNothing)
    cv2.createTrackbar("sx", "img", sx, int(rows), doNothing)
    cv2.createTrackbar("sy", "img", sy, int(cols), doNothing)
    cv2.createTrackbar("theta", "img", theta, 360, doNothing)
    while 0xFF & cv2.waitKey(1) != ord('q'):
        xc = cv2.getTrackbarPos("xc", "img")
        yc = cv2.getTrackbarPos("yc", "img")
        sx = cv2.getTrackbarPos("sx", "img")
        sy = cv2.getTrackbarPos("sy", "img")
        theta = cv2.getTrackbarPos("theta", "img")
        img = create2DGaussian(rows, cols, xc, yc, sx, sy, theta)
        cv2.imshow('img', cv2.applyColorMap(scaleImage2_uchar(img),
                                            cv2.COLORMAP_JET))
    cv2.destroyAllWindows()

def DFT(input):
    # %% The Discrete Fourier Transform - Part I - Obtaining real and imaginary
    # parts of the Fourier Transform
    img = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Plane 0 - Real", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Plane 1 - Imaginary", cv2.WINDOW_KEEPRATIO)

    planes = [np.zeros(img.shape, dtype=np.float64),
              np.zeros(img.shape, dtype=np.float64)]
    planes[0][:] = np.float64(img[:])

    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)

    planes = cv2.split(img2)

    # cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
    # cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

    while 0xFF & cv2.waitKey(1) != ord('q'):
        cv2.imshow('Original', img)
        cv2.imshow('Plane 0 - Real', planes[0])
        cv2.imshow('Plane 1 - Imaginary', planes[1])
    cv2.destroyAllWindows()

def DFT2(input):
    # %% DFT - Part II -> Applying the log transform
    img = cv2.imread('img/rectangle.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Plane 0 - Real", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Plane 1 - Imaginary", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Mag", cv2.WINDOW_KEEPRATIO)

    planes = [np.zeros(img.shape, dtype=np.float64),
              np.zeros(img.shape, dtype=np.float64)]

    planes[0][:] = np.float64(img[:])
    planes[1][:] = np.float64(img[:])

    cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
    cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)

    mag = cv2.magnitude(planes[0], planes[1])
    mag += 1
    mag = np.log(mag)

    cv2.normalize(mag, mag, 1, 0, cv2.NORM_MINMAX)

    while cv2.waitKey(1) != ord('q'):
        cv2.imshow('Original', img)
        cv2.imshow('Plane 0 - Real', planes[0])
        cv2.imshow('Plane 1 - Imaginary', planes[1])
        cv2.imshow('Mag', mag)
    cv2.destroyAllWindows()

def DFT3(input):
    # %% DFT - Part III -> Shifting the Transform
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Mag", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Mag Shifted", cv2.WINDOW_KEEPRATIO)

    planes = [np.zeros(img.shape, dtype=np.float64),
              np.zeros(img.shape, dtype=np.float64)]

    planes[0][:] = np.float64(img[:])
    planes[1][:] = np.float64(img[:])
    cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
    cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)

    mag = cv2.magnitude(planes[0], planes[1])
    mag += 1
    mag = np.log(mag)

    cv2.normalize(mag, mag, 1, 0, cv2.NORM_MINMAX)

    while cv2.waitKey(1) != ord('q'):
        # print(mag)
        cv2.imshow('Original', img)
        cv2.imshow('Mag Shifted', np.fft.fftshift(mag))
        cv2.imshow('Mag', mag)
    cv2.destroyAllWindows()

def DFTTRUE(input):
    # %% The Discrete Fourier Transform
    rows = 200
    cols = 200
    disk = np.zeros((rows, cols), np.float32)

    cv2.namedWindow("disk", cv2.WINDOW_KEEPRATIO)

    xc = 100
    yc = 100
    radius = 20

    cv2.createTrackbar("xc", "disk", xc, disk.shape[0], doNothing)
    cv2.createTrackbar("yc", "disk", yc, disk.shape[1], doNothing)
    cv2.createTrackbar("radius", "disk", radius, int(disk.shape[1] / 2), doNothing)

    while cv2.waitKey(1) != ord('q'):
        xc = cv2.getTrackbarPos("xc", "disk")
        yc = cv2.getTrackbarPos("yc", "disk")
        radius = cv2.getTrackbarPos("radius", "disk")
        disk = createWhiteDisk2(200, 200, xc, yc, radius)

        cv2.imshow("disk", disk)
    cv2.destroyAllWindows()

def lowpassFilter(input):
    # %% The Discrete Fourier Transform - Part III - Lowpass Filtering
    # Ressaltar o surgimento de "falseamento", isto é, frequencia notáveis
    # quando é feita a transformada inversa. Isto ocorre pq o filtro é IDEAL.
    # Comparar o resultado da filtragem usando uma Gaussiana como filtro.
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

    radius = 50
    cv2.namedWindow("mask", cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar("radius", "mask", radius, img.shape[0], doNothing)

    while cv2.waitKey(1) != ord('q'):
        radius = cv2.getTrackbarPos("radius", "mask")

        #    mask = createWhiteDisk2(img.shape[0],
        #                            img.shape[1],
        #                            int(img.shape[0] / 2),
        #                            int(img.shape[1] / 2),
        #                            radius)
        mask = create2DGaussian(img.shape[0],
                                img.shape[1],
                                int(img.shape[0] / 2),
                                int(img.shape[1] / 2),
                                radius,
                                radius,
                                theta=0)

        img = np.float32(img)

        planes = [img, np.zeros(img.shape, dtype=np.float32)]

        img2 = cv2.merge(planes)
        img2 = cv2.dft(img2)
        planes = cv2.split(img2)

        planes[0] = np.multiply(np.fft.fftshift(mask), planes[0])
        planes[1] = np.multiply(np.fft.fftshift(mask), planes[1])
        img2 = cv2.merge(planes)
        img2 = cv2.idft(img2)
        img2 = np.fft.fftshift(img2)

        cv2.imshow("img", scaleImage2_uchar(img))
        cv2.imshow("planes_0", np.fft.fftshift(planes[0]))
        cv2.imshow("planes_1", np.fft.fftshift(planes[1]))
        cv2.imshow("mask", np.fft.fftshift(mask))
        cv2.imshow("img2", np.fft.fftshift(scaleImage2_uchar(img2[:, :, 1])))
    cv2.destroyAllWindows()

def highpassFilter(input):
    # %% The Discrete Fourier Transform - Part IV - Highpass Filtering
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    radius = 50
    cv2.namedWindow("mask", cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar("radius", "mask", radius, img.shape[0], doNothing)

    while cv2.waitKey(1) != ord('q'):
        radius = cv2.getTrackbarPos("radius", "mask")

        #    mask = createWhiteDisk2(img.shape[0],
        #                            img.shape[1],
        #                            int(img.shape[0] / 2),
        #                            int(img.shape[1] / 2),
        #                            radius)
        mask = 1.0 - create2DGaussian(img.shape[0],
                                      img.shape[1],
                                      int(img.shape[0] / 2),
                                      int(img.shape[1] / 2),
                                      radius + 1,
                                      radius + 1,
                                      theta=0)

        img = np.float32(img)

        planes = [img, np.zeros(img.shape, dtype=np.float32)]

        img2 = cv2.merge(planes)
        img2 = cv2.dft(img2)
        planes = cv2.split(img2)

        planes[0] = np.multiply(np.fft.fftshift(mask), planes[0])
        planes[1] = np.multiply(np.fft.fftshift(mask), planes[1])
        img2 = cv2.merge(planes)
        img2 = cv2.idft(img2)
        img2 = np.fft.fftshift(img2)

        cv2.imshow("img", scaleImage2_uchar(img))
        cv2.imshow("planes_0", np.fft.fftshift(planes[0]))
        cv2.imshow("planes_1", np.fft.fftshift(planes[1]))
        cv2.imshow("mask", np.fft.fftshift(mask))
        cv2.imshow("img2", np.fft.fftshift(scaleImage2_uchar(img2[:, :, 1])))
    cv2.destroyAllWindows()

def DFT4(input):
    # %% The Discrete Fourier Transform - Visualizing sinusoidal images - Part II
    rows = 250
    cols = 250
    freq = 1
    theta = 2

    cv2.namedWindow("mag", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

    cv2.createTrackbar("Freq", "img", freq, 500, doNothing)
    cv2.createTrackbar("Theta", "img", theta, 100, doNothing)

    while cv2.waitKey(1) != ord('q'):
        freq = cv2.getTrackbarPos("Freq", "img")
        theta = cv2.getTrackbarPos("Theta", "img")

        img = createCosineImage2(rows, cols, float(freq / 1e3), theta)
        img3 = np.copy(img)
        planes = [img3, np.zeros(img3.shape, np.float64)]
        img2 = cv2.merge(planes)
        img2 = cv2.dft(img2)
        planes = cv2.split(img2)
        mag = cv2.magnitude(planes[0], planes[1])
        mag = applyLogTransform(mag)

        cv2.imshow("img", cv2.applyColorMap(scaleImage2_uchar(img),
                                            cv2.COLORMAP_JET))
        cv2.imshow("mag", cv2.applyColorMap(np.fft.fftshift(scaleImage2_uchar(mag)),
                                            cv2.COLORMAP_JET))
    cv2.destroyAllWindows()

def DFTAddNoise(input):
    # %% The Discrete Fourier Transform - Adding sinusoidal noise to images - Part I
    cv2.namedWindow("mag", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

    img = np.float32(img)
    img = img / 255.0

    rows = img.shape[0]
    cols = img.shape[1]

    freq = 90
    theta = 10
    gain = 30

    cv2.createTrackbar("Freq", "img", freq, 500, doNothing)
    cv2.createTrackbar("Theta", "img", theta, 100, doNothing)
    cv2.createTrackbar("Gain", "img", gain, 100, doNothing)

    while cv2.waitKey(1) != ord('q'):
        freq = cv2.getTrackbarPos("Freq", "img")
        theta = cv2.getTrackbarPos("Theta", "img")
        gain = cv2.getTrackbarPos("Gain", "img")

        noise = createCosineImage2(rows, cols, float(freq / 1e3), theta)
        noise = img + float(gain / 100.0) * noise

        img3 = np.copy(noise)
        planes = [img3, np.zeros(img3.shape, np.float64)]
        img2 = cv2.merge(planes)
        img2 = cv2.dft(img2)
        planes = cv2.split(img2)
        mag = cv2.magnitude(planes[0], planes[1])
        mag = applyLogTransform(mag)

        cv2.imshow("img", scaleImage2_uchar(noise))
        cv2.imshow("mag", cv2.applyColorMap(
            np.fft.fftshift(scaleImage2_uchar(mag)),
            cv2.COLORMAP_OCEAN))
    cv2.destroyAllWindows()

def DFTAddNoise2(input):
    # %% The Discrete Fourier Transform - Adding sinusoidal noise to images - Part II
    cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("mask", cv2.WINDOW_KEEPRATIO)

    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)
    img = img / 255.0;

    rows = img.shape[0]
    cols = img.shape[1]

    freq = 90
    theta = 10
    gain = 30

    cv2.createTrackbar("Freq", "img", freq, 500, doNothing)
    cv2.createTrackbar("Theta", "img", theta, 100, doNothing)
    cv2.createTrackbar("Gain", "img", gain, 100, doNothing)

    bandwidth = 2
    outer_radius = 256 - 210 + bandwidth
    inner_radius = 256 - 210 - bandwidth
    cv2.createTrackbar("in_radius", "mask", inner_radius, img.shape[1], doNothing)
    cv2.createTrackbar("out_radius", "mask", outer_radius, img.shape[1], doNothing)

    while cv2.waitKey(1) != ord('q'):
        freq = cv2.getTrackbarPos("Freq", "img")
        theta = cv2.getTrackbarPos("Theta", "img")
        gain = cv2.getTrackbarPos("Gain", "img")

        outer_radius = cv2.getTrackbarPos("in_radius", "mask")
        inner_radius = cv2.getTrackbarPos("out_radius", "mask")

        noise = img + float(gain / 100.0) * createCosineImage2(
            rows, cols, float(freq / 1e3), theta)

        mask = 1 - (createWhiteDisk2(rows, cols, int(cols / 2),
                                     int(rows / 2), outer_radius) - createWhiteDisk2(rows, cols, int(cols / 2),
                                                                                     int(rows / 2), inner_radius))

        planes = [np.copy(noise), np.zeros(noise.shape, np.float64)]
        img2 = cv2.merge(planes)
        img2 = cv2.dft(img2)
        planes = cv2.split(img2)
        mag = cv2.magnitude(planes[0], planes[1])
        mag = applyLogTransform(mag)
        planes[0] = np.multiply(np.fft.fftshift(mask), planes[0])
        planes[1] = np.multiply(np.fft.fftshift(mask), planes[1])
        tmp = cv2.merge(planes)
        tmp = cv2.idft(tmp)

        cv2.imshow("img", scaleImage2_uchar(noise))
        cv2.imshow("mag", cv2.applyColorMap(np.fft.fftshift(scaleImage2_uchar(mag)), cv2.COLORMAP_OCEAN))
        cv2.imshow("mask", scaleImage2_uchar(mask))
        cv2.imshow("tmp", scaleImage2_uchar(tmp[:, :, 0]))
    cv2.destroyAllWindows()

def avgBlur(input):
    #    Average Blurring

    img = cv2.imread('img/lena.png')

    cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("New", cv2.WINDOW_KEEPRATIO)

    ksizex = 0;
    ksizey = 0

    cv2.createTrackbar("ksizex", "New", ksizex, 63, doNothing)
    cv2.createTrackbar("ksizey", "New", ksizey, 63, doNothing)

    img2 = np.zeros(img.shape, dtype=np.float64)

    while cv2.waitKey(1) != ord('q'):

        ksizey = cv2.getTrackbarPos("ksizey", "New")
        ksizex = cv2.getTrackbarPos("ksizex", "New")

        if ksizex < 1:
            ksizex = 1
        if ksizey < 1:
            ksizey = 1

        img2 = cv2.blur(img, (ksizex, ksizey), img2, (-1, -1), cv2.BORDER_DEFAULT)

        cv2.imshow("Original", img)
        cv2.imshow("New", img2)

def saltAndClean(input):
    #Adding salt & pepper noise to an image and cleaning it using the median
    img = cv2.imread("img/lena.png", cv2.IMREAD_GRAYSCALE)
    
    noise = np.zeros(img.shape, np.uint8)
    img2 = np.zeros(img.shape, np.uint8)
    img3 = np.zeros(img.shape, np.uint8)
    salt = np.zeros(img.shape, np.uint8)
    pepper = np.zeros(img.shape, np.uint8)
    
    ksize = 0
    amount = 5
    cv2.namedWindow("img3", cv2.WINDOW_KEEPRATIO);
    cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO);
    cv2.createTrackbar("ksize", "img3", ksize, 15, doNothing)
    cv2.createTrackbar("amount", "img2", amount, 120, doNothing)
    
    cv2.randu(noise, 0, 255)
    
    while cv2.waitKey(1) != ord('q'):
        amount = cv2.getTrackbarPos("amount", "img2")
        ksize = cv2.getTrackbarPos("ksize", "img3")
    
        img2 = np.copy(img)
    
        salt = noise > 255 - amount
        pepper = noise < amount
    
        img2[salt == True] = 255
        img2[pepper == True] = 0
    
        img3 = cv2.medianBlur(img2, (ksize + 1) * 2 - 1)
    
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        cv2.imshow("img3", img3)


def main():
    print("Welcome to PDI tools 0.1!")
    #piceWiseTransform("lena.png")
    #logTransform("lolo.jpeg")
    #intensityTransform("lolo.jpeg")
    #FDO3("eye.jpeg")
    #LaplacOP("eye.jpeg")
    #gaussFilter("eye.jpeg")
    #DFT("img/pollen.jpg")
    #DFT2("eye.jpeg")
    #DFT3("eye.jpeg")
    #DFTTRUE("eye.jpeg")
    lowpassFilter("eye.jpeg")
    #highpassFilter("eye.jpeg")
    #DFTAddNoise2("eye.jpeg")
    #DFT4("eyes.jpeg")
    #LaplacOP2("eyes.jpeg")
if __name__ == "__main__":
    main()
    #cv2.namedWindow("img")

import random
import sys
from PIL import Image
import cv2
import numpy as np
from math import inf
import time

# --- GLOBAL DECLARATIONS --- #

img72 = cv2.imread('images/OriginalImages/image072.png')
img72padding = cv2.imread('images/OriginalImages/image072padding.png')
img92 = cv2.imread('images/OriginalImages/image092.png')
height, width, channels = img72.shape
grayImg72 = cv2.cvtColor(img72, cv2.COLOR_RGB2YCrCb)[:, :, 0]
grayImg92 = cv2.cvtColor(img92, cv2.COLOR_RGB2YCrCb)[:, :, 0]
newGrayImg72 = cv2.cvtColor(img72padding, cv2.COLOR_RGB2YCrCb)[:, :, 0]
#
newImg = np.zeros((height, width), dtype=np.uint8)

boxSize = 16
blocks92 = []
blocks72 = []


# --- DEFs -- #
def MSE(bloc1, bloc2):
    block1, block2 = np.array(bloc1), np.array(bloc2)
    return np.square(np.subtract(block1, block2)).mean()


def dichotomicSearch(bloc, pointi, pointj, minmse):
    global y, x, bloc72
    stepToVosin = 32
    debut_i = pointi - stepToVosin - 8 + 64
    debut_j = pointj - stepToVosin - 8 + 64
    while stepToVosin >= 1:
        # image 72
        fin = stepToVosin * 2 + boxSize
        for newi in range(int(debut_i), int(debut_i) + int(fin), int(stepToVosin)):
            for newj in range(int(debut_j), int(debut_j) + int(fin), int(stepToVosin)):
                bloc72 = newGrayImg72[newi:newi + boxSize, newj:newj + boxSize]
                loss = MSE(bloc, bloc72)

                if loss < minmse:
                    minmse = loss
                    x = newi
                    y = newj
                    # nkemlou m le bloc with smallest mse
        stepToVosin /= 2
        debut_i = x
        debut_j = y
    return x, y, minmse, bloc72


# a discuter je suis pas sur
def drawRectanglesForIchotomicSearch(img1, blocks1, img2, blocks2):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    for i in range(len(blocks1)):
        cv2.rectangle(img1, (blocks1[i][0], blocks1[i][1]),
                      (blocks1[i][0] + boxSize, blocks1[i][1] + boxSize), (r, g, b), 2)

    for i in range(len(blocks2)):
        cv2.rectangle(img2, (blocks2[i][0], blocks2[i][1]),
                      (blocks2[i][0] + boxSize, blocks2[i][1] + boxSize), (r, g, b), 2)


def main():
    # image 92
    for i in range(0, height - boxSize, boxSize):  # colonneImage
        for j in range(0, width - boxSize, boxSize):  # ligneImage
            bloc92 = grayImg92[i:i + boxSize, j:j + boxSize]
            MINmse = inf
            x, y, minmse, bloc = dichotomicSearch(bloc92, i + 8, j + 8, MINmse)  # +8 centre du bloc
            if minmse > 50:
                # lahna naffectiw lbloc with mse > 50 resultant lel image resultante taena
                blocRes = bloc92 - bloc
                newImg[i:i + boxSize, j:j + boxSize] = blocRes
                blocks92.append((y, x))
                blocks72.append((j, i))
            else:
                print(f'MIN: {minmse}')


# --- MAIN --- #

if __name__ == '__main__':
    start_time = time.time()
    main()
    # drawRectanglesForIchotomicSearch(img92, blocks92, img72, blocks72)
    time = time.time() - start_time
    print(f"{time} seconds")
    imgResu = Image.fromarray(newImg)
    imgResu.save('images/dichotomicSearchResults/imageResuDichoto.png')
    # cv2.imwrite('images/dichotomicSearchResults/image072withRedRect.png', img72)  # u can use just cv2.show()
    # cv2.imwrite('images/dichotomicSearchResults/image092withGreenRect.png', img92)
    sys.exit()

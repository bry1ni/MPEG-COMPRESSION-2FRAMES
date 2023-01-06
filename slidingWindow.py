import random
import sys

from PIL import Image
import cv2
import numpy as np
from math import inf
import time

# --- GLOBAL DECLARATIONS --- #

img72 = cv2.imread('images/OriginalImages/image072.png')
img92 = cv2.imread('images/OriginalImages/image092.png')
height, width, channels = img72.shape
grayImg72 = cv2.cvtColor(img72, cv2.COLOR_RGB2YCrCb)[:, :, 0]
grayImg92 = cv2.cvtColor(img92, cv2.COLOR_RGB2YCrCb)[:, :, 0]

newImg = np.zeros((height, width), dtype=np.uint8)  # creating an empty matrix

boxSize = 16
greens = []
reds = []


# --- DEFs -- #
def MSE(bloc1, bloc2):
    block1, block2 = np.array(bloc1), np.array(bloc2)
    return np.square(np.subtract(block1, block2)).mean()


def slidingWindow(padding):
    global x, y, blocVert
    for i in range(0, height - boxSize, boxSize):  # colonneImage
        for j in range(0, width - boxSize, boxSize):  # ligneImage
            blocRouge = grayImg72[i:i + boxSize, j:j + boxSize]
            MIN = inf
            for i1 in range(max(0, i - padding), min(i + padding, height - boxSize)):
                for j1 in range(max(0, j - padding), min(j + padding, width - boxSize)):
                    blocVert = grayImg92[i1:i1 + boxSize, j1:j1 + boxSize]
                    loss = MSE(blocRouge, blocVert)
                    if loss < MIN:
                        MIN = loss
                        x = i1
                        y = j1

            if MIN > 50:
                greens.append((y, x))
                reds.append((j, i))
                print(f'found and inserted. MIN: {MIN}')
                newImg[x:x + boxSize, y:y + boxSize] = blocVert  # adding to the matrix
    return newImg, greens, reds


def drawRectanglesForSlidingWindow(img1, greens, img2, reds):
    for i in range(len(greens)):
        cv2.rectangle(img1, (greens[i][0], greens[i][1]),
                      (greens[i][0] + boxSize, greens[i][1] + boxSize), (0, 255, 0), 2)

    for i in range(len(reds)):
        cv2.rectangle(img2, (reds[i][0], reds[i][1]),
                      (reds[i][0] + boxSize, reds[i][1] + boxSize), (0, 0, 255), 2)


# --- MAIN --- #

start_time = time.time()
print('starting...')
padding = 7  # ps: padding bigger = longer time
ourImage, greensR, redsR = slidingWindow(padding)
drawRectanglesForSlidingWindow(img92, greensR, img72, redsR)
print('finish.')
time = time.time() - start_time
print(f"{time} seconds")
imgResu = Image.fromarray(ourImage)  # converting the matrix into image
imgResu.save('images/SlidingWindowResults/imageResuSlidingWindow.png')
cv2.imwrite('images/SlidingWindowResults/image072withRedRect.png', img72)  # u can use just cv2.show()
cv2.imwrite('images/SlidingWindowResults/image092withGreenRect.png', img92)  # u can use just cv2.show()
sys.exit()

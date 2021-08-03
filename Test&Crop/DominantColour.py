import cv2
import numpy as np
from skimage import io

class ColourInImage:

    img = ""
    palette = ""
    counts = ""
    def __init__(self):
        self.img = io.imread('F:\PYTHON\Projekte\FindTheDog\BETA\CropedImg\Crop.png')


    def getAvgColour(self):
        average = self.img.mean(axis=0).mean(axis=0)
        return average

    def getDominantColour(self):
        pixels = np.float32(self.img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, self.palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, self.counts = np.unique(labels, return_counts=True)
        dominant = self.palette[np.argmax(self.counts)]
        return dominant

    def plotColour(self):
        import matplotlib.pyplot as plt
        avg_patch = np.ones(shape=self.img.shape, dtype=np.uint8) * np.uint8(self.getDominantColour())

        indices = np.argsort(self.counts)[::-1]
        freqs = np.cumsum(np.hstack([[0], self.counts[indices] / float(self.counts.sum())]))
        rows = np.int_(self.img.shape[0] * freqs)

        dom_patch = np.zeros(shape=self.img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(self.palette[indices[i]])

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        ax0.imshow(avg_patch)
        ax0.set_title('Average color')
        ax0.axis('on')
        ax1.imshow(dom_patch)
        ax1.set_title('Dominant colors')
        ax1.axis('on')
        plt.show()


#dom = ColourInImage()
#print(dom.getAvgColour())
#print(dom.getDominantColour())
#dom.getDominantColour()
#dom.plotColour()

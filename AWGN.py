"""
Adds White Gaussian Noise (AWGN) to an image based on a given sigma value.

author: RV
"""

import random
from PIL import Image


SIGMA = 100 # apply sigma here


class AWGN:
    """Applies additive white Gaussian noise to a grayscale image.

    objects:
        sigma - Standard deviation of the Gaussian noise
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self._rng = random.Random()

    def add_noise(self, im):
        """Add Gaussian noise to an image.

        parameters:
            im - grayscale pixel values in [0, 255].

        Returns:
            2D list of floats with Gaussian noise added. Pixel values are
            NOT clipped so the user can decide how to handle out of range
            values.
        """
        noisy = []
        for row in im:
            noisy_row = []
            for pixel in row:
                noisy_row.append(pixel + self._rng.gauss(0.0, self.sigma))
            noisy.append(noisy_row)
        return noisy
    
def main():
    im = Image.open("citroen_GC.jpg")
    im = im.convert("L")
    name = "citroen"

    M, N = im.size

    image = []
    for v in range(N):
        row = []
        for u in range(M):
            row.append(float(im.getpixel((u, v))))
        image.append(row)
    
    noisy = AWGN(SIGMA).add_noise(image)

    out = Image.new("L", (M, N))

    out_pixel = out.load()

    for v in range(N):
        for u in range(M):
            val = noisy[v][u]
            if val < 0.0:
                val = 0.0
                out_pixel[u,v] = 0
            elif val > 255:
                val = 255
                out_pixel[u,v] = 255
            else:
                out_pixel[u,v] = int(val)   

    out.save(f"citroen_noisy_{SIGMA}.jpg")

    return

if __name__ == "__main__":
    main()


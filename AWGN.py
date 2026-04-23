"""
addeds White Gaussian Noise (AWGN) utility to an inputed image based on a given sigma value.

author: RV
"""

import random
from PIL import Image

#==================================
SIGMA = 100 # apply sigma here!!! 
#==================================

class AWGN:
    """Applies additive white Gaussian noise to a grayscale image.

    Args:
        sigma: Standard deviation of the Gaussian noise (same units as
               pixel values, i.e. [0, 255] scale).
        seed:  Optional integer random seed for reproducibility.
    """

    def __init__(self, sigma: float, seed: int = None):
        self.sigma = sigma
        self._rng = random.Random(seed)

    def apply(self, image):
        """Add Gaussian noise to an image.

        Args:
            image: 2D sequence (list of lists or numpy array) of grayscale
                   pixel values in [0, 255].

        Returns:
            2D list of floats with Gaussian noise added. Pixel values are
            NOT clipped so the caller can decide how to handle out-of-range
            values.
        """
        noisy = []
        for row in image:
            noisy_row = []
            for pixel in row:
                noisy_row.append(pixel + self._rng.gauss(0.0, self.sigma))
            noisy.append(noisy_row)
        return noisy
    
def main():
    im = Image.open("citroen_GC.jpg")
    im = im.convert("L")

    M, N = im.size

    image = []
    for v in range(N):
        row = []
        for u in range(M):
            row.append(float(im.getpixel((u, v))))
        image.append(row)
    
    noisy = AWGN(SIGMA).apply(image)

    out = Image.new("L", (M, N))

    out_px = out.load()

    for v in range(N):
        for u in range(M):
            val = noisy[v][u]
            if val < 0.0:
                val = 0.0
            elif val > 255.0:
                val = 255.0
            out_px[u,v] = int(val)

    out.save(f"Citroen_awgn_{SIGMA}.jpg")

    return

if __name__ == "__main__":
    main()


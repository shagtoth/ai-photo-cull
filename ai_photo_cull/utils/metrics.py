import cv2
import numpy as np


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def exposure_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    blacks = hist[0]
    whites = hist[-1]
    clipped = (blacks + whites) / hist.sum()
    return float(1.0 - clipped)


def noise_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.Laplacian(blur, cv2.CV_64F).var()
    return float(min(noise / 500.0, 1.0))


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    high_freq = np.sum(magnitude > np.percentile(magnitude, 95))
    return float(min(high_freq / 50000.0, 1.0))

import numpy as np

def gradient(im):
    # Sobel operator
    op1 = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    op2 = np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]])
    kernel1 = np.zeros(im.shape)
    kernel1[:op1.shape[0], :op1.shape[1]] = op1
    kernel1 = np.fft.fft2(kernel1)

    kernel2 = np.zeros(im.shape)
    kernel2[:op2.shape[0], :op2.shape[1]] = op2
    kernel2 = np.fft.fft2(kernel2)

    fim = np.fft.fft2(im)
    Gx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)
    Gy = np.real(np.fft.ifft2(kernel2 * fim)).astype(float)

    G = np.sqrt(Gx**2 + Gy**2)
    Theta = np.arctan(Gy, Gx) * 180 / np.pi
    return G, Theta    

def nonMaximumSuppression(image):
    det, phase = gradient(image)
    gmax = np.zeros(det.shape)
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            if phase[i][j] < 0:
                phase[i][j] += 360

            if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
            # 0 degrees
                if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
                    if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                        gmax[i][j] = det[i][j]
            # 45 degrees
                if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                    if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
                        gmax[i][j] = det[i][j]
            # 90 degrees
                if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                    if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
                        gmax[i][j] = det[i][j]
            # 135 degrees
                if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                    if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
                        gmax[i][j] = det[i][j]
    return gmax
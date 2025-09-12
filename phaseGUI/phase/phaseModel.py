from scipy.fft import fft2, ifft2, fftshift
import numpy as np


class PhaseModel:   
    def __init__(self):
        pass
    def get_mask(self, shape, rho1, rho2, phaseshift=None, m=None):
        M, N = shape[0], shape[1]
        y, x = np.ogrid[:M, :N]
        cx, cy = M//2, N//2
        fx, fy = x - cx, y - cy
        r = np.sqrt(fx**2 + fy**2)
        phi = np.arctan2(fy, fx)
        
        mask = np.zeros((M, N), dtype=complex)
        
        if phaseshift is not None:
            mask[(r <= rho1)] = np.exp(1j * phaseshift)
            mask[(r > rho1) & (r <= rho2)] = 1.0
        elif m is not None:
            mask = ((r >= rho1) & (r <= rho2)) * np.exp(1j * m * phi)
        else:
            mask = (r >= rho1) & (r <= rho2)
        return mask

    def normalize_output(self, arr):
        return arr
        arr = np.asarray(arr, dtype=np.float32)
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            arr_norm = 255 * (arr - min_val) / (max_val - min_val)
        else:
            arr_norm = np.zeros_like(arr)
        return arr_norm.astype(np.uint8)

    def darkfield(self, I, rho1, rho2):
        M = self.get_mask(I.shape, rho1, rho2)
        out = abs(ifft2( fftshift(fft2(np.sqrt(I))) * M ))**2

        out_norm = self.normalize_output(out)
        return out_norm, M

    def phase_shifted_darkfield(self, I, rho1, rho2, phaseshift):
        Mps = self.get_mask(I.shape, rho1, rho2, phaseshift=phaseshift)
        out = abs(ifft2( fftshift(fft2(np.sqrt(I))) * Mps ))**2
        out_norm = self.normalize_output(out)
        return out_norm, Mps

    def intensity_weighted_darkfield(self, I, rho1, rho2, alpha):
        out_d, M = self.darkfield(I, rho1, rho2)
        out = (1+alpha*I)*out_d
        out_norm = self.normalize_output(out)
        return out_norm, M

    def spiral_darkfield(self, I, rho1, rho2, m):
        Msd = self.get_mask(I.shape, rho1, rho2, m=m)
        out = abs(ifft2(fftshift(fft2(np.sqrt(I))) * Msd)) ** 2
        out_norm = self.normalize_output(out)
        return out_norm, Msd


if __name__ == "__main__":
    import cv2
    I = np.random.rand(512, 512).astype(np.float32)
    I = cv2.imread("C:/Users/julia/Documents/!SLO/AngioSLO_Measurement_2Deg_NFrames_150_2023-03-15_12-48-21-4080/channel1/calib/_calib_00000.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    pm = PhaseModel()
    o, m = pm.darkfield(I, 1, 256)
    print(o.max(), o.min())
    o, m = pm.phase_shifted_darkfield(I, 1, 256, np.pi/2)
    print(o.max(), o.min())
    o, m = pm.intensity_weighted_darkfield(I, 1, 256, 0.2)
    print(o.max(), o.min())
    o, m = pm.spiral_darkfield(I, 0, 256, 1)
    print(o.max(), o.min())

    # import matplotlib.pyplot as plt
    # plt.imshow(o, cmap='gray')
    # plt.show()
    # plt.hist(o.ravel(), 256, [0,256])
    # plt.show()


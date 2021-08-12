#%%
import cv2
from scipy import signal
import numpy as np
from PIL import Image
#%%
def ssim(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))

    #Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    ux = signal.convolve2d(gray1, gaussian_kernel, boundary='symm', mode='same')
    uy = signal.convolve2d(gray2, gaussian_kernel, boundary='symm', mode='same')

    uxx = signal.convolve2d(gray1 * gray1, gaussian_kernel, boundary='symm', mode='same')
    uyy = signal.convolve2d(gray2 * gray2, gaussian_kernel, boundary='symm', mode='same')
    uxy = signal.convolve2d(gray1 * gray2, gaussian_kernel, boundary='symm', mode='same')


    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    # Refer to  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
    #           (2004). Image quality assessment: From error visibility to
    #           structural similarity. IEEE Transactions on Image Processing,
    #           13, 600-612.

    k1, k2 = 0.01, 0.03
    min, max = np.iinfo(gray1.dtype.type).min, np.iinfo(gray1.dtype.type).max
    range = max - min
    C1, C2 = (k1 * range) ** 2, (k2 * range) ** 2

    # 简化上公式: SSIM = (A * B) / (C * D)

    A = 2 * ux * uy + C1
    B = 2 * vxy + C2
    C = ux ** 2 + uy ** 2 + C1
    D = vx + vy + C2

    SSIM = (A * B) / (C * D)
    # If we need ssim index, compute it mean:
    ssim_index = np.mean(SSIM)

    msk = Image.fromarray(np.uint8(SSIM * 255), 'L')
    msk.save('img/mask.png')
    diff = cv2.imread('img/mask.png', cv2.IMREAD_GRAYSCALE)

    # # 提取轮廓
    thresh = cv2.threshold(diff, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(cnts))

    cnts_need = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # gray1.size = gray1.width * gray1.height * gray1.channel
        # gray has only 1 channel, So area is equal with gray1.size.
        if area > gray1.size * 0.05:
            cnts_need.append(cnt)

    # Box:
    for cnt in cnts_need:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(gray1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(gray2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    # Edge:
    point_size = 1
    point_color = (255, 255, 255) # BGR
    thickness = 0 # 可以為 0 、4、8

    for cnt in cnts_need:
        for point in cnt:
            cv2.circle(gray2, tuple(point[0]), point_size, point_color, thickness)

    bg = np.zeros((gray2.shape[0], gray2.shape[1]), np.uint8)
    # Get mask one by one:
    for cnt in cnts_need:
        mask = cv2.fillPoly(bg, [cnt], (255,255,255))
        res = cv2.bitwise_and(img2, img2, mask=mask)
        cv2.imshow('Result', res)
        k = cv2.waitKey(0)
        if k == 27 or k == ord('q'):
            cv2.destroyAllWindows()
            
    # Get all masks:    
    masks = cv2.fillPoly(bg, cnts_need, (255,255,255))
    res = cv2.bitwise_and(img2, img2, mask=masks)
    cv2.imshow('Result', res)
    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):
        cv2.destroyAllWindows()

    return ssim_index, diff
#%%
pth1, pth2 = 'img/071_054_001.png', 'img/071_054_001_DF.png'

img1 = cv2.imread(pth2) # RGB [...,::-1]
img2 = cv2.imread(pth1)

score, diff = ssim(img1, img2)


# %%

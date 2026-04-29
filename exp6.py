from scipy.signal import wiener
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Kanpur.jpg', 0)
noise = 20 * np.sin(np.linspace(0,50,img.shape[1]))
noise = noise.reshape(1,-1)
noisy = img + noise
noisy = np.clip(noisy,0,255).astype(np.uint8)

wiener_img = wiener(noisy)

plt.imshow(wiener_img, cmap='gray')
plt.title('Wiener Filtered Image')
plt.axis('off')
plt.show()

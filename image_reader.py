import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2


def read_images(file):
    with open(file, 'rb') as f:
        dictionary = pickle.load(f, encoding='latin1')
    return dictionary


images = read_images(r'D:\Facultate\Licenta\test_batch')

X = images["data"]
Y = images['labels']

X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
Y = np.array(Y)

index = 0

for image in X:
    cv2.imwrite(r'D:\Facultate\Licenta\GuiApp\QueryImages' + str(index) + ".jpg", image)
    index += 1

# fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
# for j in range(5):
#     for k in range(5):
#         i = np.random.choice(range(len(X)))
#         axes1[j][k].set_axis_off()
#         axes1[j][k].imshow(X[i:i+1][0], interpolation='nearest')

#fig.show()
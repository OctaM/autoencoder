import numpy as np
from keras.models import Model
from keras.datasets import cifar10
import cv2
from sklearn.metrics import label_ranking_average_precision_score
from keras.models import load_model
import time

print('Loading cifar10 dataset')
t0 = time.time()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
t1 = time.time()
print('cifar 10 dataset loaded in: ', t1-t0)

print('Loading model :')
t0 = time.time()
autoencoder = load_model(r'.\neural_networks\cifar10_with_dataaug_batchnorm_normalVers.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
t1 = time.time()
print('Model loaded in: ', t1-t0)

scores = []


def retrieve_closest_elements(test_code, test_label, learned_codes):
    distances = []
    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(y_train).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    if sum(distances.shape) != 0:
        labels = labels.reshape(distances.shape)
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    return sorted_distances, sorted_labels, sorted_indexes


def compute_average_precision_score(test_codes, test_labels, learned_codes, n_samples):
    out_labels = []
    out_distances = []
    retrieved_elements_indexes = []
    for i in range(len(test_codes)):
        sorted_distances, sorted_labels, sorted_indexes = retrieve_closest_elements(test_codes[i], test_labels[i], learned_codes)
        out_distances.append(sorted_distances[:n_samples])
        out_labels.append(sorted_labels[:n_samples])
        retrieved_elements_indexes.append(sorted_indexes[:n_samples])

    out_labels = np.array(out_labels)
    out_labels_file_name = 'E:/Licenta/GuiApp/computed_data/out_labels_{}'.format(n_samples)
    np.save(out_labels_file_name, out_labels)

    out_distances_file_name = 'E:/Licenta/GuiApp/computed_data/out_distances_{}'.format(n_samples)
    out_distances = np.array(out_distances)
    np.save(out_distances_file_name, out_distances)
    score = label_ranking_average_precision_score(out_labels, out_distances)
    scores.append(score)
    return score


def retrieve_closest_images(test_element, test_label, n_samples=10):
    #test_element = x_test[test_element]
    #print(test_element)
    test_label = y_test[test_label]
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                          learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])

    test_code = encoder.predict(np.array([test_element]))
    test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])

    distances = []

    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(y_train).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    labels = labels.reshape(distances.shape)
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    kept_indexes = sorted_indexes[:n_samples]

    score = label_ranking_average_precision_score(np.array([sorted_labels[:n_samples]]), np.array([sorted_distances[:n_samples]]))

    print("Average precision ranking score for tested element is {}".format(score))

    #original_image = x_test[70]
    original_image = test_element
    retrieved_images_labels = []
    # cv2.imshow('original_image', original_image)
    #retrieved_images = []
    retrieved_images = x_train[int(kept_indexes[0]), :]
    for i in range(1, n_samples):
        retrieved_images = np.hstack((retrieved_images, x_train[int(kept_indexes[i]), :]))
        #retrieved_images.append(x_train[int(kept_indexes[i]), :])
    for i in range(0, n_samples):
        retrieved_images_labels.append(y_train[int(kept_indexes[i]), :])
    print("Retrieved labels:")
    labels = []
    for label in retrieved_images_labels:
        if label[0] == 9:
            print("truck")
            labels.append("truck")
        elif label[0] == 0:
            print("airplane")
            labels.append("airplane")
        elif label[0] == 1:
            print("automobile")
            labels.append("auto")
        elif label[0] == 2:
            print("bird")
            labels.append("bird")
        elif label[0] == 3:
            print("cat")
            labels.append("cat")
        elif label[0] == 4:
            print("deer")
            labels.append("deer")
        elif label[0] == 5:
            print("dog")
            labels.append("dog")
        elif label[0] == 6:
            print("frog")
            labels.append("frog")
        elif label[0] == 7:
            print("horse")
            labels.append("horse")
        elif label[0] == 8:
            print("ship")
            labels.append("ship")
        else:
            print(label[0])
    # cv2.imshow('Results', retrieved_images)
    # cv2.waitKey(0)

    # cv2.imwrite('E:/Facultate/unsupervises-image-retrieval/test_results_64v3/original_image5.jpg', 255 * cv2.resize(original_image, (0,0), fx=3, fy=3))
    # cv2.imwrite('E:/Facultate/unsupervises-image-retrieval/test_results_64v3/retrieved_results5.jpg', 255 * cv2.resize(retrieved_images, (0,0), fx=2, fy=2))
    return (255 * cv2.resize(original_image, (0, 0), fx=3, fy=3), 255 * cv2.resize(retrieved_images, (0, 0), fx=2, fy=2), score, labels)


def test_model(n_test_samples, n_train_samples):
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0], learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])
    test_codes = encoder.predict(x_test)
    test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])
    indexes = np.arange(len(y_test))
    np.random.shuffle(indexes)
    indexes = indexes[:n_test_samples]

    print('Start computing score for {} train samples'.format(n_train_samples))
    t1 = time.time()
    score = compute_average_precision_score(test_codes[indexes], y_test[indexes], learned_codes, n_train_samples)
    t2 = time.time()
    print('Score computed in: ', t2-t1)
    print('Model score:', score)


# def plot_denoised_images():
#     denoised_images = autoencoder.predict(x_test_noisy.reshape(x_test_noisy.shape[0], x_test_noisy.shape[1], x_test_noisy.shape[2], 1))
#     test_img = x_test_noisy[2]
#     resized_test_img = cv2.resize(test_img, (280, 280))
#     cv2.imshow('input', resized_test_img)
#     cv2.waitKey(0)
#     output = denoised_images[0]
#     resized_output = cv2.resize(output, (280, 280))
#     cv2.imshow('output', resized_output)
#     cv2.waitKey(0)
#     cv2.imwrite('E:/Facultate/unsupervises-image-retrieval/test_results/noisy_image2.jpg', 255 * resized_test_img)
#     cv2.imwrite('E:/Facultate/unsupervises-image-retrieval/test_results/denoised_image2.jpg', 255 * resized_output)


# To test the whole model
n_test_samples = 1000
# n_train_samples = [10, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
#                    20000, 30000, 40000, 50000, 60000]
n_train_samples = [10]

# for n_train_sample in n_train_samples:
#     test_model(n_test_samples, n_train_sample)
#
# np.save('E:/Facultate/unsupervises-image-retrieval/computed_data/scores', np.array(scores))


# To retrieve closest image
# print('y_train = {}'.format(y_test[65]))
# retrieve_closest_images(65, 65)


# To plot a denoised image
# plot_denoised_images()


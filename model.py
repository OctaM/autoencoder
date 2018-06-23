import numpy as np
import time
import cv2
import pickle
import os
import const

from sklearn.metrics import label_ranking_average_precision_score
from keras.models import load_model
from keras.models import Model
from keras.datasets import cifar10
from pathlib import Path


class MyModel:
    def __init__(self, file_path):
        self.x_train = []
        self.y_train = []
        self.x_test = None
        self.y_test = None
        self.noise_factor = 0.5
        self.x_train_noisy = None
        self.x_test_noisy = None
        self.learned_codes = []
        self._load_data()
        self._load_autoencoder(file_path)
        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('encoder').output)
        self.scores = []
        #
        self._create_codes()
        self._load_codes()

    def _load_data(self):
        t0 = time.time()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # reads labels from txt file in the same order as the images
        # with open('cifar_labels.txt', 'r') as labels_file:
        #     for label in labels_file:
        #         self.y_train.append(int(label))
        #     self.y_train = np.array(self.y_train)
        #
        # # loads images from train directory
        # for image_name in os.listdir(os.path.join(const.train_dir)):
        #     self.x_train.append(np.array(cv2.imread(os.path.join(const.train_dir, image_name), cv2.IMREAD_UNCHANGED),
        #                                  dtype=float))

        #self.x_train = np.array(self.x_train)
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        self.x_train = np.reshape(self.x_train, (len(self.x_train), 32, 32, 3))
        self.x_test = np.reshape(self.x_test, (len(self.x_test), 32, 32, 3))

        self.x_train_noisy = self.x_train + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=self.x_train.shape)
        self.x_test_noisy = self.x_test + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=self.x_test.shape)
        t1 = time.time()
        print('Data loaded in: {}'.format(t1 - t0))

    def _load_autoencoder(self, file_path):
        t0 = time.time()
        self.autoencoder = load_model(file_path)
        t1 = time.time()
        print("Model loaded in: {}".format(t1 - t0))

    def _create_codes(self):

        learned_codes = Path('learned_codes.pkl')
        if not learned_codes.exists():
            with open('learned_codes.pkl', 'ab') as file:
            #     for image in os.listdir(os.path.join(const.train_dir)):
            #         image = np.array(cv2.imread(os.path.join(const.train_dir, image),
            #                                     cv2.IMREAD_UNCHANGED), dtype=float)
            #         image = np.asarray(image) / 255.0
            #         image = np.expand_dims(image, axis=0)
            #         self.learned_codes.append(self.encoder.predict(image))
            #
            #     self.learned_codes = np.array(self.learned_codes)
            #     self.learned_codes.reshape(self.learned_codes.shape[0], self.learned_codes.shape[2],
            #                                self.learned_codes.shape[3], self.learned_codes.shape[4])
                self.learned_codes = self.encoder.predict(self.x_train)
                pickle.dump(self.learned_codes, file)

    def _load_codes(self):
        try:
            with open('learned_codes.pkl', 'rb') as file:
                self.learned_codes = pickle.load(file)
                # self.learned_codes = np.array(self.learned_codes)
                # self.learned_codes = self.learned_codes.reshape(self.learned_codes.shape[0], self.learned_codes.shape[2],
                #                                                 self.learned_codes.shape[3], self.learned_codes.shape[4])
        except FileNotFoundError:
            print("Learned codes file is missing")

    def retrieve_closest_elements(self, test_code, test_label, learned_codes):
        distances = []
        for code in learned_codes:
            distance = np.linalg.norm(code - test_code)
            distances.append(distance)
        nb_elements = learned_codes.shape[0]
        distances = np.array(distances)
        learned_code_index = np.arange(nb_elements)
        labels = np.copy(self.y_train).astype('float32')
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

    def compute_average_precision_score(self, test_codes, test_labels, learned_codes, n_samples):
        out_labels = []
        out_distances = []
        retrieved_elements_indexes = []
        for i in range(len(test_codes)):
            sorted_distances, sorted_labels, sorted_indexes = self.retrieve_closest_elements(test_codes[i], test_labels[i],
                                                                                        learned_codes)
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
        self.scores.append(score)
        return score

    def retrieve_closest_images(self, test_element, test_label, n_samples=10, binary_signatures=0):
        test_label = self.y_test[test_label]
        #self.learned_codes = self.encoder.predict(self.x_train)
        if len(self.learned_codes.shape) != 2:
            self.learned_codes = self.learned_codes.reshape(self.learned_codes.shape[0],
                                                            self.learned_codes.shape[1] * self.learned_codes.shape[2] *
                                                            self.learned_codes.shape[3])

        test_code = self.encoder.predict(np.array([test_element]))
        test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])
        # test_code = (test_code-np.min(test_code)) / (np.max(test_code) - np.min(test_code))
        if binary_signatures == 1:
            test_code[test_code > np.mean(test_code)] = 1
            test_code[test_code <= np.mean(test_code)] = 0
        distances = []

        for code in self.learned_codes:
            # code = (code - np.min(code)) / (np.max(code) - np.min(code))
            if binary_signatures == 1:
                code[code > np.mean(test_code)] = 1
                code[code <= np.mean(test_code)] = 0
            #distance = np.linalg.norm(code - test_code)
            distance = np.count_nonzero(code != test_code)
            distances.append(distance)
        nb_elements = self.learned_codes.shape[0]
        distances = np.array(distances)
        learned_code_index = np.arange(nb_elements)
        labels = np.copy(self.y_train).astype('float32')
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

        score = label_ranking_average_precision_score(np.array([sorted_labels[:n_samples]]),
                                                      np.array([sorted_distances[:n_samples]]))

        print("Average precision ranking score for tested element is {}".format(score))

        # original_image = x_test[70]
        original_image = test_element
        retrieved_images_labels = []
        # cv2.imshow('original_image', original_image)
        retrieved_images = []
        retrieved_images2 = []
        retrieved_images2= self.x_train[int(kept_indexes[0]), :]
        retrieved_images.append(self.x_train[int(kept_indexes[0]), :])
        for i in range(1, n_samples):
            retrieved_images2 = np.hstack((retrieved_images2, self.x_train[int(kept_indexes[i]), :]))
            retrieved_images.append(self.x_train[int(kept_indexes[i]), :])
        for i in range(0, n_samples):
            retrieved_images_labels.append(self.y_train[int(kept_indexes[i])])
        print("Retrieved labels:")
        labels = []
        for label in retrieved_images_labels:
            if label == 9:
                print("truck")
                labels.append("truck")
            elif label == 0:
                print("airplane")
                labels.append("airplane")
            elif label == 1:
                print("automobile")
                labels.append("auto")
            elif label == 2:
                print("bird")
                labels.append("bird")
            elif label == 3:
                print("cat")
                labels.append("cat")
            elif label == 4:
                print("deer")
                labels.append("deer")
            elif label == 5:
                print("dog")
                labels.append("dog")
            elif label == 6:
                print("frog")
                labels.append("frog")
            elif label == 7:
                print("horse")
                labels.append("horse")
            elif label == 8:
                print("ship")
                labels.append("ship")
            else:
                print(label)
        # return (255 * cv2.resize(original_image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
        #         255 * cv2.resize(retrieved_images, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC), score, labels)
        cv2.imwrite('./result.jpg', 255 * cv2.resize(retrieved_images2, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        return (255 * cv2.resize(original_image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                retrieved_images, score, labels)

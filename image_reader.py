import xml.etree.ElementTree as et
import os
import cv2
import numpy as np

path_to_xml = r'E:\Licenta\Licenta\Resources\Annotations'
path_to_images = r'E:\Licenta\Licenta\Resources\pasc_data\validation'


def get_pascalvoc_labels_from_xml(path_to_xml, path_to_images):
    with open('./test_labels.txt', 'a') as labels_file:
        for file in os.listdir(path_to_images):
            file_name, _ = file.split('.')
            file_name = file_name + '.xml'
            xml_file = et.parse(os.path.join(path_to_xml, file_name)).getroot()
            for child in xml_file.iter('name'):
                labels_file.write(child.text + '\n')
                break


def get_unique_labels():
    labels = []
    with open('./train_labels.txt', 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if line not in labels:
                labels.append(line)

    print(len(labels))


def get_cifar_data(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def plot_images_to_folder():
    image_number = 10000
    for data_batch in range(2, 6):
        dict = get_cifar_data(r'E:\Licenta\Licenta\Resources\cifar\cifar-10-batches-py\data_batch_' + str(data_batch))
        for image in dict[b'data']:
            im_r = im_r = image[0:1024].reshape(32, 32)
            im_g = image[1024:2048].reshape(32, 32)
            im_b = image[2048:].reshape(32, 32)
            img = np.dstack((im_b, im_g, im_r))
            cv2.imwrite(r'./TrainImages/' + str(image_number) + '.jpg', img)
            image_number += 1


plot_images_to_folder()

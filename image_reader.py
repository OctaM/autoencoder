import xml.etree.ElementTree as et
import os

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


get_unique_labels()

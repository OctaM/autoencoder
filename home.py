from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow import variable_scope

from model import MyModel

import cv2
import os
import pickle
import numpy as np
import const


class AppWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.query_img = None
        self.file_path = const.model_path
        self.canvas = Canvas(root, width=1280, height=720)
        self.canvas.pack(expand=YES, fill=BOTH)
        self.init_window()
        self.model = MyModel(self.file_path)
        self.kept_images = []
        self.images_container = []
        self._create_labels()
        self.binary_signatures = IntVar()
        self.nn_arhitectures = {
            'cifar',
            'mnist',
            'fmnist'
        }

    # create Label objects for displaying retrieved images
    def _create_labels(self):

        for placed_images in range(10):
            label = Label(self.canvas)
            # label.config(image=img)
            self.images_container.append(label)
            if placed_images % 3 == 0:
                placed_images += 1
                const.image_offset_y += 120
                const.image_offset_x = 768
                label.place(x=const.image_offset_x, y=const.image_offset_y)
            else:
                placed_images += 1
                const.image_offset_x += 120
                label.place(x=const.image_offset_x, y=const.image_offset_y)

    def open_explorer(self):
        ftypes = [('jpg files', '*.jpg'), ('png files', '*.png'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        for index in range(10):
            self.canvas.delete('label' + str(index))
            self.canvas.delete('image' + str(index))

        for image in self.images_container:
            image.config(image="")
            
        self.canvas.delete('retr_imgs')

        if fl is not None:
            self.read_query_image(fl)
            image_to_display = cv2.resize(self.query_img, (100, 100), interpolation=cv2.INTER_CUBIC)
            b,g,r = cv2.split(image_to_display)
            img = cv2.merge((r,g,b))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            #self.canvas = Label(self, image=img)
            self.canvas.create_image(0, 20, image=img, anchor=NW)
            self.canvas.img = img
            self.kept_images.append(img)
            # self.canvas.pack()
            # self.canvas.place(x=30, y=30)

    def load_query_images(self, file_path):
        with open(file_path, 'rb') as f:
            dictionary = pickle.load(f, encoding='latin1')

        images_to_display = []
        images = dictionary["data"]
        labels = dictionary['labels']
        images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        labels = np.array(labels)

        for image in images:
            b,g,r = cv2.split(image)
            img = cv2.merge((r,g,b))
            img = Image.fromarray(img)
            images_to_display.append(ImageTk.PhotoImage(image=img))

        novi = Toplevel()
        canvas = Canvas(novi, width=800, height=800)
        canvas.pack(expand=YES, fill=BOTH)
        index = 0
        for i in range(0, 600, 40):
            for j in range(0, 600, 40):
                if index < 5:
                    canvas.create_image(i, j, image=images_to_display[index], anchor=NW)
                    index += 1

    def read_query_image(self, image_path):
        if os.path.exists(image_path):
            try:
                self.query_img = cv2.imread(image_path)
            except ValueError:
                print("Cannot read image")

    def _set_binary_signatures(self, binary_sign):
        if binary_sign.get() == 0:
            self.binary_signatures = 0
        else:
            self.binary_signatures = 1

    def retrieve_similar_images(self):
        original_image, retrieved_images, score, labels = self.model.retrieve_closest_images(self.query_img.astype('float32') / 255., 70, 10,  self.binary_sign.get())
        cv2.imwrite(r'./tmp/original_img.jpg', original_image)

        for retr_image, image_container in zip(retrieved_images, self.images_container):
            retr_image = 255 * cv2.resize(retr_image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            b, g, r = cv2.split(retr_image)
            img = cv2.merge((r, g, b))
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            image_container.config(image=img)
            image_container.image = img


        original_image = cv2.imread(r'./tmp/original_img.jpg')
        self.canvas.create_text(0, 0, text="This is the query image", anchor=NW)
        # self.canvas.create_image(0, 20, image=original_imgtk, anchor=NW)
        index = 0
        for label in labels:

            if index == 0:
                self.canvas.create_text(140, 100, text=label, anchor=NW, tag='label' + str(index))
                index += 1
            else:
                self.canvas.create_text(140+63*index, 100, text=label, anchor=NW, tag='label' + str(index))
                index += 1

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Image retrieval")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        open_explorer = Button(self.canvas, text="Open image", command=self.open_explorer, width=30, height=2)
        start_retrieving = Button(self.canvas, text="Find images", command=self.retrieve_similar_images, width=30, height=2)
        # placing the button on my window
        open_explorer.place(x=250, y=250)
        start_retrieving.place(x=250, y=300)
        nn_arhitectures = {
            'cifar',
            'mnist',
            'fmnist'
        }
        variable = StringVar(self.canvas)
        variable.set('cifar')

        drop_down = OptionMenu(self.canvas, variable, *nn_arhitectures)
        drop_down.place(x=250, y=350)

        self.binary_sign = IntVar()

        checkbutton = Checkbutton(self.canvas, text="Binarize signatures", variable=self.binary_sign, onvalue=1,
                                  offvalue=0)
        checkbutton.place(x=500, y=305)


if __name__ == '__main__':
    root = Tk()
    root.geometry("1280x720")
    app = AppWindow(root)
    root.mainloop()

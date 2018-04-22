from tkinter import *
import cv2
import os
from tkinter import filedialog
from test_model import retrieve_closest_images
from PIL import Image, ImageTk
import pickle
import numpy as np


class AppWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.query_img = None
        self.init_window()

    def open_explorer(self):
        ftypes = [('jpg files', '*.jpg'), ('png files', '*.png'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl is not None:
            self.read_query_image(fl)
            image_to_display = cv2.resize(self.query_img, (100, 100), interpolation=cv2.INTER_CUBIC)
            b,g,r = cv2.split(image_to_display)
            img = cv2.merge((r,g,b))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            canvas = Label(self, image=img)
            canvas.photo = img
            canvas.pack()
            canvas.place(x=30, y=30)

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

    def retrieve_similar_images(self):
        original_image, retrieved_images, score, labels = retrieve_closest_images(self.query_img.astype('float32') / 255., 70)
        #print()
        cv2.imwrite(r'./tmp/original_img.jpg', original_image)
        cv2.imwrite(r'./tmp/retrieved_images.jpg', retrieved_images)

        original_image = cv2.imread(r'./tmp/original_img.jpg')
        retrieved_images = cv2.imread(r'./tmp/retrieved_images.jpg')

        b, g, r = cv2.split(original_image)
        img = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        original_imgtk = ImageTk.PhotoImage(image=im)
        b, g, r = cv2.split(retrieved_images)
        img = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        retrieved_imgtk = ImageTk.PhotoImage(image=im)
        novi = Toplevel()
        canvas = Canvas(novi, width=800, height=800)
        canvas.pack(expand=YES, fill=BOTH)
        canvas.create_text(0, 0, text="This is the query image", anchor=NW)
        canvas.create_image(0, 20, image=original_imgtk, anchor=NW)
        index = 0
        for label in labels:

            if index == 0:
                canvas.create_text(140, 100, text=label, anchor=NW)
                index += 1
            else:
                canvas.create_text(140+63*index, 100, text=label, anchor=NW)
                index += 1
        canvas.create_image(130, 0, image=retrieved_imgtk, anchor=NW)
        #canvas.text="This is a text"
        canvas.original_imgtk = original_imgtk
        canvas.retrieved_imgtk = retrieved_imgtk

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Image retrieval")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        open_explorer = Button(self, text="Open image", command=self.open_explorer)
        start_retrieving = Button(self, text="Find images", command=self.retrieve_similar_images)
        # placing the button on my window
        open_explorer.place(x=0, y=0)
        start_retrieving.place(x=200, y=100)

if __name__ == '__main__':
    root = Tk()
    root.geometry("640x480")
    app = AppWindow(root)
    root.mainloop()

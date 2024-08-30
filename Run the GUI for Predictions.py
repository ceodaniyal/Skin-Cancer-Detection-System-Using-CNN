import tkinter as tk
import numpy as np
import cv2
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('skin_cancer_detection.h5')

Categories = ['benign', 'malignant']

# Initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Skin Cancer Detection')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (175, 175))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_data = np.array(image).reshape(-1, 175, 175, 3)
    pred = model.predict(image_data)
    sign = Categories[int(pred[0][0])]
    label.configure(text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        pass

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Skin Cancer Detection", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()

from tkinter import *
import tkinter as tk
import PIL
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings


labelNames = "0123456789"
labelNames = [l for l in labelNames]

model = keras.models.load_model('model.h5')

warnings.filterwarnings('ignore')


def reset():
    img=ImageDraw.Draw(image1)
    img.rectangle([(0, 0), (280, 280)], fill="black")
    cv.delete("all")
    T.delete(1.0, 'end')



def predict():
    image = image1.resize((28, 28))
    image = np.asarray(image)
    image_ = image.reshape((-1, 784))
    image = np.array(image_, dtype='float32')
    predict = model.predict(image)
    predict_ = tf.nn.softmax(predict).numpy()
    prediction = np.argmax(predict_)
    print(prediction)
    T.insert(tk.END, prediction)


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=15, fill='white', smooth=TRUE, capstyle=ROUND)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='white', width=15)
    lastx, lasty = x, y


root = Tk()

lastx, lasty = None, None
image_number = 0


cv = Canvas(root, width=280, height=280, bg='black')
# --- PIL
image1 = PIL.Image.new('L', (280, 280), 'black')
draw = ImageDraw.Draw(image1)
cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)


T= Text(root, height=1, width = 5, font= ('Arial', 16, 'bold'))
l = Label(root, text = "Prediction")
l.config(font=("Courier", 14))


btn_predict = Button(text="predict", command=predict)
btn_reset = Button(text="reset", command=reset)


l.pack()
T.pack()
btn_predict.pack()
btn_reset.pack()

root.mainloop()

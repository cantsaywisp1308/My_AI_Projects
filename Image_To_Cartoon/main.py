#import libraries
import cv2
import imutils
import easygui
import imageio
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import dialog
from tkinter import *
from PIL import ImageTk, Image

def upload():
    ImagePath = easygui.fileopenbox()
    cartoonify(ImagePath)

def cartoonify(ImagePath):
    originalImage = cv2.imread(ImagePath)
    originalImage = cv2.cvtColor(originalImage,cv2.COLOR_BGR2RGB)
    #check if image is chosen
    if originalImage is None:
        print("Cannot choose any image. Please choose a file")
        sys.exit()
    Resized1 = cv2.resize(originalImage, (960,540))
    grayScaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    Resized2 = cv2.resize(grayScaleImage, (960, 540))

    #applying median blur to smoothen ur image
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    Resized3 = cv2.resize(smoothGrayScale, (960,540))

    #retrieving edges for cartoon effect
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    Resized4 = cv2.resize(getEdge, (960, 540))

    #applying bilateral filter to remove noise and keep sharpe edges as required
    colorImage = cv2.bilateralFilter(originalImage, 9, 300, 300)
    Resized5 = cv2.resize(colorImage, (960, 540))

    #masking edge image with BEAUTIFY Image
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    Resized6 = cv2.resize(cartoonImage, (960, 540))

    #Plotting the whole transition
    images = [Resized1, Resized2, Resized3, Resized4, Resized5, Resized6]
    fig ,axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks': [], 'yticks' :[]},
                             gridspec_kw= dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
    plt.show()

# For the main window where we upload the image. When we run the code, a new window will pop up
# that gives the option to upload the image from the desktop. Here we set the geometry like
# the size of the window, and the title of the window which we set to ” Cartoonify Your Image !”
# and then set the labeling style.

top = Tk()
top.geometry('500x500')
top.title('Cartoonify your images !')
top.configure(background='#eadbc8')
label = Label(top, background='#eadbc8', font=('ariel',20,'bold'))

upload = Button(top, text='Cartoonify an Image', command=upload, padx=50, pady=50)
upload.configure(background='blue', foreground='white', font=('ariel',10,'bold'))
upload.pack(side=TOP, pady=200)

save1 = Button(top, text="Save cartoon image", command= lambda: save(Resized6, ImagePath), padx=30, pady=5)
save1.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
save1.pack(side=TOP, pady=50)

def save(Resized6, ImagePath):
    #saving file using imwrite()
    newName = "Cartoonified Image"
    path1 = os.path.dirname(ImagePath)
    extension = os.path.splitext(ImagePath)[1]
    path = os.path.join((path1, newName+extension))
    cv2.imwrite(path, cv2.cvtColor(Resized6, cv2.COLOR_BGR2RGB))
    I = "Image saved by name " + newName + "at " + path
    tk.messagebox.showinfo(title=None, message = I)



top.mainloop()

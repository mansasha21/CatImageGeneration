from tkinter import *
from tkinter.messagebox import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# класс Paint
class Paint(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.img = np.ones((256, 256, 1), dtype=np.float32)
        self.parent = parent
        # Параметры кисти по умолчанию
        self.brush_size = 1
        self.brush_color = "black"
        self.color = "black"
        self.model = torch.load('C:\\Users\\Alexandr\\Documents\\mansasha\\study\\projects\\CatImageGeneration\\models\\catgen_99.zip').cuda()

        # Устанавливаем компоненты UI
        self.setUI()

    # Метод рисования на холсте
    def clear(self):
        self.img = np.ones((256, 256, 1), dtype=np.float32)
        self.canv.delete("all")

    def draw(self, event):
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)

        self.img = cv2.circle(self.img, (event.x, event.y), 2, (0, 0, 0), thickness=-1)

        # Изменение цвета кисти

    def set_color(self, new_color):
        self.color = new_color

        # Изменение размера кисти

    def set_brush_size(self, new_size):
        self.brush_size = new_size

    def setUI(self):
        # Устанавливаем название окна
        self.parent.title("Demo Paint")
        # Размещаем активные элементы на родительском окне
        self.pack(fill=BOTH, expand=1)

        self.columnconfigure(2, weight=1)
        self.rowconfigure(2, weight=1)

        # Создаем холст с белым фоном
        self.canv = Canvas(self, bg="white")

        # Приклепляем канвас методом grid. Он будет находится в 3м ряду, первой колонке,
        # и будет занимать 7 колонок, задаем отступы по X и Y в 5 пикселей, и
        # заставляем растягиваться при растягивании всего окна

        self.canv.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=E + W + S + N)

        # задаем реакцию холста на нажатие левой кнопки мыши
        self.canv.bind("<B1-Motion>", self.draw)

        clear_btn = Button(self, text="Очистить", width=10, command=self.clear)
        clear_btn.grid(row=0, column=0, sticky=W)

        one_btn = Button(self, text="transform", width=10, command=self.plot_img)
        one_btn.grid(row=1, column=0)
        one_btn = Button(self, text="save", width=10, command=lambda: self.set_brush_size(2))
        one_btn.grid(row=0, column=0)

    def plot_img(self):
        img = ((cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB) - 0.5) / 0.5).transpose((2, 0, 1))
        res = self.model(torch.tensor([img]).cuda())[0]
        plt.imshow(res.cpu().detach().numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
        plt.show()






# выход из программы
def close_win():
    if askyesno("Выход", "Вы уверены?"):
        root.destroy()

# вывод справки
def about():
    showinfo("Demo Paint", "Простейшая рисовалка от сайта: https://it-black.ru")

# функция для создания главного окна
def main():
    global root
    root = Tk()
    root.geometry("256x256+300+300")
    app = Paint(root)
    m = Menu(root)
    root.config(menu=m)

    fm = Menu(m)
    m.add_cascade(label="Файл", menu=fm)
    fm.add_command(label="Выход", command=close_win)

    hm = Menu(m)
    m.add_cascade(label="Справка", menu=hm)
    hm.add_command(label="О программе", command=about)
    root.mainloop()

if __name__ == "__main__":
    main()
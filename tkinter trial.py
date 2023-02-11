import tkinter as tk

root = tk.Tk()

label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

def button_click():
    label.config(text="Button clicked!")

button = tk.Button(root, text="Click me!", command=button_click)
button.pack()

root.mainloop()

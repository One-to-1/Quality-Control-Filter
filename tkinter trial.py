import tkinter as tk

class Installer:
    def __init__(self, master):
        self.master = master
        self.pages = []
        self.value = None

        self.create_widgets()

    def create_widgets(self):
        # Create page 1
        page1 = tk.Frame(self.master)
        tk.Label(page1, text="Welcome to my app!").pack()
        tk.Button(page1, text="Next", command=self.show_page2).pack()
        self.pages.append(page1)

        # Create page 2
        page2 = tk.Frame(self.master)
        tk.Label(page2, text="Enter a value:").pack()
        self.entry = tk.Entry(page2)
        self.entry.pack()
        tk.Button(page2, text="Back", command=self.show_page1).pack()
        tk.Button(page2, text="Next", command=lambda: self.show_page3(self.entry.get())).pack()
        self.pages.append(page2)

        # Create page 3
        page3 = tk.Frame(self.master)
        tk.Label(page3, text="Do you want to pass the value to function 2 or function 3?").pack()
        tk.Button(page3, text="Function 2", command=lambda: self.function2(self.value)).pack()
        tk.Button(page3, text="Function 3", command=lambda: self.function3(self.value)).pack()
        tk.Button(page3, text="Back", command=self.show_page2).pack()
        self.pages.append(page3)

        # Show the first page
        self.show_page1()

    def show_page1(self):
        self.pages[0].pack()
        self.pages[1].pack_forget()
        self.pages[2].pack_forget()

    def show_page2(self):
        self.pages[0].pack_forget()
        self.pages[1].pack()
        self.pages[2].pack_forget()

    def show_page3(self, value):
        self.value = value
        self.pages[0].pack_forget()
        self.pages[1].pack_forget()
        self.pages[2].pack()

    def function2(self, value):
        print("Function 2 received value:", value)

    def function3(self, value):
        print("Function 3 received value:", value)

root = tk.Tk()
installer = Installer(root)
root.mainloop()

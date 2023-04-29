import customtkinter as ctk

def button_callback():
    print("Button clicked")

app = ctk.CTk()
app.title("Snoop")
app.geometry("500x300")
button = ctk.CTkButton(app, text="my button", command=button_callback)
button.grid(row=0, column=0, padx=20, pady=20)

app.mainloop()

# Step 7. Build the GUI using Python’s Tkinter library

import random
import tkinter as tk
from tkinter import *

root = tk.Tk()
filename = "Chat Bot"
root.title(f"Chat Bot")
root.geometry('500x400')
root.resizable(False, False)
message = tk.StringVar()

chat_win = Frame(root, bd = 1, bg = 'white', width = 50, height = 8)
chat_win.place(x = 6, y = 6, height = 300, width = 488)

textcon = tk.Text(chat_win, bd = 1, bg = 'white', width = 50, height = 8)
textcon.pack(fill = "both", expand = True)

mes_win = Entry(root, width = 30, xscrollcommand = True, textvariable = message)
mes_win.place(x = 6, y = 310, height = 60, width = 380)
mes_win.focus()

textcon.config(fg = 'black')
textcon.tag_config('usr', foreground = 'black')
textcon.insert(END, "Bot: Welcome to the G4 Metrobus Experience.\n\n")
mssg = mes_win.get()

exit_list = ['exit', 'break', 'quit', 'see you later', 'chat with you later', 'end the chat', 'bye', 'ok bye']

def greet_res(text):
    text = text.lower()
    bot_greet = ['hi', 'hello', 'hola', 'hey', 'howdy']
    usr_greet = ['hi', 'hello', 'hola', 'hey', 'howdy', 'greetings', 'wassup', 'whats up', 'wasgood']
    for word in text.split():
        if word in usr_greet:
            return random.choice(bot_greet)

def send_msz(event = None):
    usr_input = message.get()
    usr_input = usr_input.lower()
    textcon.insert(END, f'you: {usr_input}' + '\n', 'usr')
    if usr_input in exit_list:
        textcon.config(fg = 'black')
        textcon.insert(END, "Bot: Ok bye! I hope you have had a great experience!\n")
        return root.destroy()
    else:
        textcon.config(fg = 'black')
        if greet_res(usr_input) != None:
            lab = f"Bot: {greet_res(usr_input)}" + '\n'
            textcon.insert(END, lab)
            mes_win.delete(0, END)

button_send = Button(root, text = 'Send', bg = 'dark green', activebackground = 'grey', command = send_msz, width = 12,
                     height = 5, font = ('Arial'))
button_send.place(x = 376, y = 310, height = 60, width = 110)
root.bind('<Return>', send_msz, button_send)
root.mainloop()

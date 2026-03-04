from tkinter import *
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter.ttk import Progressbar
import threading
from functions import *


def process_batch(paths):
    images, videos = collect_files(paths)
    total = len(images) + len(videos)
    if total == 0:
        return

    progress["value"] = 0
    step = 100 / total
    done = 0
    tasks = []

    
    for img in images:
        tasks.append(("image", img))
    for vid in videos:
        tasks.append(("video", vid))

    for kind, path in tasks:
        if kind == "image":
            process_image(str(path))
        elif kind == "video":
            process_video(str(path))

        done += 1
        progress["value"] = done * step
        progress.update_idletasks()

    progress["value"] = 100
    progress.update_idletasks()


def dropper(event):
    raw = app.tk.splitlist(event.data)
    paths = [p.strip("{}") for p in raw]

    threading.Thread(
        target=process_batch,
        args=(paths,),
        daemon=True
    ).start()


#ui
app = TkinterDnD.Tk()
app.geometry("1280x720")
app.title("Quality detector")
app.config(bg="#333333")

label = Label(
    app,
    text="YOLO image/video quality detector",
    bg="#4472B6",
    fg="white",
    font=("Arial", 18),
    width=40,
    height=5,
    relief="solid",
    borderwidth=2
)
label.pack(expand=True, fill="both")
label.drop_target_register(DND_FILES)
label.dnd_bind("<<Drop>>", dropper)


progress = Progressbar(app, orient="horizontal", length=600, mode="determinate")
progress.pack(pady=15)

def execute_project():
    app.mainloop()
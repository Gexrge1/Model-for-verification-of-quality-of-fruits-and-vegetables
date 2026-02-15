from tkinter import *
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter.ttk import Progressbar
import threading
from functions import *


#there is no video analysis for NIR images
def process_batch(paths, is_nir=False):
    images, videos = collect_files(paths)
    total = len(images) + len(videos) if not is_nir else len(images)

    progress["value"] = 0
    step = 100 / total if total > 0 else 0
    done = 0
    tasks = []

    if is_nir:
        for img in images:
            tasks.append(("nir_image", img))
    else:
        for img in images:
            tasks.append(("image", img))
        for vid in videos:
            tasks.append(("video", vid))

    for kind, path in tasks:
        if kind == "image":
            process_image(str(path))
        elif kind == "video":
            process_video(str(path))
        elif kind == "nir_image":
            process_nir_image(str(path))

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
        args=(paths, False),
        daemon=True
    ).start()

def dropper_nir(event):
    raw = app.tk.splitlist(event.data)
    paths = [p.strip("{}") for p in raw]

    threading.Thread(
        target=process_batch,
        args=(paths, True),
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

nir_label = Label(
    app,
    text="NIR images quality detector",
    bg="#924177",
    fg="white",
    font=("Arial", 18),
    width=40,
    height=5,
    relief="solid",
    borderwidth=2
)
nir_label.pack(expand=True, fill="both")
nir_label.drop_target_register(DND_FILES)
nir_label.dnd_bind("<<Drop>>", dropper_nir)

progress = Progressbar(app, orient="horizontal", length=600, mode="determinate")
progress.pack(pady=15)

def execute_project():
    app.mainloop()
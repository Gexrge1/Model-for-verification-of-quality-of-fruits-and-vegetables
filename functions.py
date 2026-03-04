from ultralytics import YOLO
from pathlib import Path
from typing import Tuple
import threading, cv2, datetime, csv, sys


def resource_path(relative):
    try:
        base = sys._MEIPASS
    except Exception:
        base = Path(__file__).parent
    return base / relative


# yolo models
detector = YOLO(resource_path(r"models\recogn-all.pt"))
quality_model = YOLO(resource_path(r"models\quality_fresh.pt"))

# constants
image_extensions = {".jpg", ".jpeg", ".png"}
video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

crops_folder = Path("crops")
crops_folder.mkdir(exist_ok=True)

timestamp_folder = None
image_out = None
video_out = None
defects_out = None
csv_file = None
csv_writer = None
csv_lock = threading.Lock()


def create_output_folders():
    global timestamp_folder, image_out, video_out, defects_out, csv_file, csv_writer

    if timestamp_folder is None:
        
        timestamp = datetime.datetime.now().strftime("%H-%M %d-%m-%Y")
        desktop = Path.home() / "Desktop"
        timestamp_folder = desktop / "QualityOutput" / timestamp
        image_out = timestamp_folder / "yolo_images"
        video_out = timestamp_folder / "yolo_videos"

        image_out.mkdir(parents=True, exist_ok=True)
        video_out.mkdir(parents=True, exist_ok=True)

        csv_file = open(timestamp_folder / "results.csv", "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "source_type",
            "source_name",
            "frame",
            "item_class",
            "confidence_score"
        ])

# csv loggs
def log_rotten(source_type, source_name, frame, cls_name, conf):
    if "rotten" not in cls_name.lower():
        return

    with csv_lock:
        csv_writer.writerow([
            source_type,source_name,
            frame,cls_name,f"{conf:.2f}"
        ])
        

def log_defects(source_type, source_name, frame, num_def):
    with csv_lock:
        csv_writer.writerow([
            source_type,source_name,
            frame,"defects",num_def
        ])
       

def process_image(file_path):
    create_output_folders()
    source_name = Path(file_path).name

    frame = cv2.imread(file_path)
    det_res = detector.predict(frame, conf=0.5, verbose=False)[0]

    for i, box in enumerate(det_res.boxes):
        conf = float(box.conf)
        if conf < 0.5:
            continue

        cls = int(box.cls)
        name = det_res.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_path = crops_folder / f"{name}_{i}.jpg"
        cv2.imwrite(str(crop_path), crop)

        qual_res = quality_model.predict(
            str(crop_path), conf=0.4, save=False, verbose=False
        )

        annotated_crop = crop.copy()  # Start with original crop for annotation

        for r in qual_res:
            for qbox in r.boxes:
                qcls = int(qbox.cls)
                qname = r.names[qcls]
                qconf = float(qbox.conf)
                log_rotten("image", source_name, "-", qname, qconf)

            annotated_crop = r.plot(img=annotated_crop)  # Plot on the crop

        out_file = image_out / f"{Path(file_path).stem}_{i}_annotated.jpg"
        cv2.imwrite(str(out_file), annotated_crop)
        print(f"Saved annotated crop to {out_file}")

    for f in crops_folder.glob("*.jpg"):
        f.unlink()


def process_video(video_path):
    create_output_folders()
    source_name = Path(video_path).name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * 1)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        det_res = detector.predict(frame, conf=0.5, verbose=False)[0]

        annotated_frame = det_res.plot()  # Plot detections on full frame

        for f in crops_folder.glob("*.jpg"):
            f.unlink()

        crop_offsets = {}
        for i, box in enumerate(det_res.boxes):
            conf = float(box.conf)
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_file = crops_folder / f"crop_{i}.jpg"
            cv2.imwrite(str(crop_file), crop)
            crop_offsets[str(crop_file)] = (x1, y1)

        crop_files = list(crops_folder.glob("*.jpg"))
        if crop_files:
            qual_res = quality_model.predict(
                str(crops_folder), conf=0.5, save=False, verbose=False
            )

            for r in qual_res:
                crop_path = str(r.path)  # Path of the crop image being processed
                if crop_path in crop_offsets:
                    offset_x, offset_y = crop_offsets[crop_path]

                    for qbox in r.boxes:
                        qcls = int(qbox.cls)
                        qname = r.names[qcls]
                        qconf = float(qbox.conf)
                        log_rotten("video", source_name, frame_count, qname, qconf)

                        # Draw quality box on full frame
                        qx1, qy1, qx2, qy2 = map(int, qbox.xyxy[0])
                        cv2.rectangle(annotated_frame, (qx1 + offset_x, qy1 + offset_y), (qx2 + offset_x, qy2 + offset_y), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{qname} {qconf:.2f}", (qx1 + offset_x, qy1 + offset_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out_file = video_out / f"{Path(video_path).stem}_frame{frame_count}.jpg"
        cv2.imwrite(str(out_file), annotated_frame)
        print(f"Saved annotated frame to {out_file}")

    cap.release()

    for f in crops_folder.glob("*.jpg"):
        f.unlink()


def collect_files(paths):
    images, videos = [], []

    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in image_extensions:
                    images.append(f)
                elif f.suffix.lower() in video_extensions:
                    videos.append(f)
        elif p.is_file():
            if p.suffix.lower() in image_extensions:
                images.append(p)
            elif p.suffix.lower() in video_extensions:
                videos.append(p)

    return images, videos


from ultralytics import YOLO
from pathlib import Path
import numpy as np
from typing import Tuple
from utils.edge import *
from utils.general import *
from utils.threshold import *
import threading, cv2, datetime, csv, sys


def resource_path(relative):
    try:
        base = sys._MEIPASS
    except Exception:
        base = Path(__file__).parent
    return base / relative


"""
Imported NIR defect detector, made by: Riccardo Spolaor.
Link to his project: https://github.com/RiccardoSpolaor/Fruit-Inspection
"""
def detect_defects(colour_image: np.ndarray, nir_image: np.ndarray, image_name: str = '', tweak_factor: float = .3,
                   sigma: float = 1., threshold_1: int = 60, threshold_2: int = 130,
                   verbose: bool = True) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function that executes the task of detecting the defects of a fruit.
    It firstly masks the fruit, then it looks for the defects.
    If the task is run in `verbose` mode, the visualization of the defect regions of the fruit is plotted.
    Parameters
    ----------
    colour_image: np.ndarray
        Colour image of the fruit whose defects have to be detected
    nir_image: np.ndarray
        Near Infra-Red image of the same fruit represented in `colour_image`
    image_name: str, optional
        Optional name of the image to visualize during the plotting operations
    tweak_factor: float, optional
        Tweak factor to apply to the "Tweaked Otsu's Algorithm" in order to obtain the binary segmentation mask
        (default: 0.3)
    sigma: float, optional
        Value of sigma to apply to the Gaussian Blur operation before the use of Canny's algorithm (default: 1)
    threshold_1: int, optional
        Value of the first threshold that is used in Canny's algorithm (default: 60)
    threshold_2: int, optional
        Value of the second threshold that is used in Canny's algorithm (default: 120)
    verbose: bool, optional
        Whether to run the function in verbose mode or not (default: True)
    Returns
    -------
    retval: int
        Number of defect regions found in the fruit
    stats: np.ndarray
        Array of statistics about each defect region:
            - The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction;
            - The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction;
            - The horizontal size of the bounding box;
            - The vertical size of the bounding box;
            - The total area (in pixels) of the defect.
    centroids: np.ndarray
        Array of centroid points about each defect region.
    defect_mask: np.ndarray
        The defect mask.
    """
    # Filter the NIR image by median blur
    f_nir_image = cv2.medianBlur(nir_image, 5)
    # Get the fruit mask through Tweaked Otsu's algorithm
    mask = get_fruit_segmentation_mask(f_nir_image, ThresholdingMethod.TWEAKED_OTSU, tweak_factor=tweak_factor)
    # Apply the mask to the filtered NIR image
    m_nir_image = apply_mask_to_image(f_nir_image, mask)
    # Get the edge mask through Gaussian Blur and Canny's method
    edge_mask = apply_gaussian_blur_and_canny(m_nir_image, sigma, threshold_1, threshold_2)
    # Erode the mask to get rid of the edges of the bound of the fruit
    erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    eroded_mask = cv2.erode(mask, erode_element)
    # Remove background edges from the edge mask
    edge_mask = apply_mask_to_image(edge_mask, eroded_mask)
    # Apply Closing operation to fill the defects according to the edges and obtain the defect mask
    close_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    defect_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, close_element)
    defect_mask = cv2.medianBlur(defect_mask, 7)
    # Perform a connected components labeling to detect the defects
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask)
    if verbose:
        print(f'Detected {retval - 1} defect{"" if retval - 1 == 1 else "s"} for image {image_name}.')
        # Get highlighted defects on the fruit
        highlighted_roi = get_highlighted_roi_by_mask(colour_image, defect_mask, 'red')
        circled_defects = np.copy(colour_image)
        for i in range(1, retval):
            s = stats[i]
            # Draw a red ellipse around the defect
            cv2.ellipse(circled_defects, center=tuple(int(c) for c in centroids[i]),
                       axes=(s[cv2.CC_STAT_WIDTH] // 2 + 10, s[cv2.CC_STAT_HEIGHT] // 2 + 10),
                       angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=3)
        plot_image_grid([highlighted_roi, circled_defects],
                        ['Detected defects ROI', 'Detected defects areas'],
                        f'Defects of the fruit {image_name}')
    return retval - 1, stats[1:], centroids[1:], defect_mask

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
        defects_out = timestamp_folder / "NIR_recognitions"

        image_out.mkdir(parents=True, exist_ok=True)
        video_out.mkdir(parents=True, exist_ok=True)
        defects_out.mkdir(parents=True, exist_ok=True)

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

def process_nir_image(file_path):
    create_output_folders()
    source_name = Path(file_path).name

    nir_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if nir_image is None:
        print(f"Failed to load NIR image: {file_path}")
        return

    # Convert grayscale to BGR for color_image
    colour_image = cv2.cvtColor(nir_image, cv2.COLOR_GRAY2BGR)

    try:
        num_def, stats, cents, defect_mask = detect_defects(colour_image, nir_image, verbose=False)
        print(f"Detected {num_def} defects for {source_name}")

        log_defects("nir_image", source_name, "-", num_def)

        annotated_image = colour_image.copy()

        if num_def > 0:
            for j in range(num_def):
                s = stats[j]
                c = cents[j]
                cv2.ellipse(annotated_image, center=tuple(int(k) for k in c),
                            axes=(s[2] // 2 + 10, s[3] // 2 + 10),
                            angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=3)

        out_file = defects_out / f"{Path(file_path).stem}_defects_ellipses.jpg"
        cv2.imwrite(str(out_file), annotated_image)
        print(f"Saved defects ellipses image to {out_file}")

        if defect_mask is not None:
            highlighted_roi = get_highlighted_roi_by_mask(colour_image, defect_mask, 'red')
            highlight_file = defects_out / f"{Path(file_path).stem}_highlighted.jpg"
            cv2.imwrite(str(highlight_file), highlighted_roi)
            print(f"Saved highlighted ROI to {highlight_file}")

    except Exception as e:
        print(f"Defect detection failed for {source_name}: {e}")

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


import argparse
import glob
import os,shutil

import cv2

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.25, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("extracting tags for each image...")
if args.images.endswith(".txt"):
    with open(args.images, "r") as myfile:
        lines = myfile.readlines()
        files = map(lambda x: os.path.join(os.path.dirname(args.images), x.strip()), lines)
else:
    files = sorted(glob.glob("%s/*.jpg" % args.images))

conf_sum = 0
detection_count = 0
if os.path.exists('export'):
    shutil.rmtree('export')
else:
    os.mkdir('export')
for file in files:
    print(file)
    mat = cv2.imread(file)

    width, height, inference_time, results = yolo.inference(mat)

    print("%s in %s seconds: %s classes found!" %
          (os.path.basename(file), round(inference_time, 2), len(results)))

    output = []

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1024, 768)
    output = []
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        conf_sum += confidence
        detection_count += 1
        # draw a bounding box rectangle and label on the image
        color = (255, 0, 255)

        print("%s with %s confidence" % (name, round(confidence, 2)))
        
        file_name = os.path.basename(file)
        files = os.path.splitext(file_name)    
        crop_img = mat[y:y+h, x:x+w]
        output.append(crop_img)

    for i, roi in enumerate(output):
        cv2.imwrite(f"export/{files[0]}_{i}_cropped.jpg", roi)

print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
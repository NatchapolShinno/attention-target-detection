import cv2
import numpy as np
import tensorflow as tf

# Load SSD model
ssd_model = tf.saved_model.load("/Users/shinno/attention-target-detection/head_detection/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/")

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (320, 320))
    image = tf.expand_dims(image, axis=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image

def postprocess_boxes(boxes, width, height):
    boxes = np.squeeze(boxes)
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    return boxes

# Load image
image = cv2.imread("head_detection/frame_input/00119121.jpg")
height, width, _ = image.shape

# Preprocess image
input_image = preprocess_image(image)

# Run inference
detections = ssd_model(input_image)

# Extract bounding boxes and class labels
boxes = detections['detection_boxes'].numpy()
classes = detections['detection_classes'].numpy()

# Filter detections for persons (assuming class label for person is 1)
person_boxes = boxes[classes == 1]

# Convert normalized boxes to image coordinates
person_boxes = postprocess_boxes(person_boxes, width, height)

# Save bounding box coordinates to a text file
output_file = "output_ssd_320.txt"
with open(output_file, "w") as f:
    for box in person_boxes:
        x_min, y_min, x_max, y_max = box
        f.write(f"{output_file}, {int(x_min)}, {int(y_min)}, {int(x_max)}, {int(y_max)}\n")

# Draw bounding boxes on the image
for box in person_boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Show the result
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

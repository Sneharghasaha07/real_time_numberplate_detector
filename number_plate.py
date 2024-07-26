import cv2
from google.cloud import vision
import io
import os

# Correct the path to your Haar cascade file
harcascade = r"D:\haldia3\model\haarcascade_russian_plate_number.xml"

# Set up Google Cloud Vision client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\sneha\Downloads\rich-access-426306-r9-a6744daa40f5.json"
client = vision.ImageAnnotatorClient()

# Create the directory to save the images if it doesn't exist
save_dir = "plates"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
plate_counter = 0  # To save multiple plates separately

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    detected_plates = []

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x: x + w]
            detected_plates.append(img_roi)
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        if detected_plates:
            for idx, img_roi in enumerate(detected_plates):
                plate_path = os.path.join(save_dir, f"scanned_img_{plate_counter}.jpg")
                cv2.imwrite(plate_path, img_roi)
                plate_counter += 1
            print("Plate images saved.")
        else:
            print("No number plate detected to save")

    elif key == ord('d'):
        for idx in range(plate_counter):
            plate_path = os.path.join(save_dir, f"scanned_img_{idx}.jpg")
            if os.path.exists(plate_path):
                with io.open(plate_path, 'rb') as image_file:
                    content = image_file.read()

                image = vision.Image(content=content)

                try:
                    response = client.text_detection(image=image)
                    texts = response.text_annotations
                    if texts:
                        plate_text = texts[0].description.strip()
                        print(f"Detected Number Plate Text ({plate_path}): {plate_text}")
                    else:
                        print(f"No text detected in {plate_path}")
                except Exception as e:
                    print(f"Error occurred during OCR: {e}")
            else:
                print(f"No image to perform OCR for {plate_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

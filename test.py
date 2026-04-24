from ultralytics import YOLO
import cv2

# ==========================
# 🔧 CONFIG
# ==========================
MODEL_PATH = "runs/detect/train2/weights/best.pt"
IMAGE_PATH = "cardboard-box-package-parcel-white-shipping-concept-65130685.jpg"
CONF = 0.5

# ==========================
# 🚀 LOAD MODEL
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# 🖼️ LOAD IMAGE
# ==========================
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("❌ Image not found")
    exit()

# ==========================
# 🔍 PREDICT
# ==========================
results = model(img, conf=CONF)

boxes = results[0].boxes
names = model.names

damaged_count = 0
undamaged_count = 0

for box in boxes:
    cls = int(box.cls[0])
    if cls == 1:
        damaged_count += 1
    else:
        undamaged_count += 1

# ==========================
# 🎨 DRAW
# ==========================
annotated = results[0].plot()

# ==========================
# 📊 PRINT
# ==========================
print(f"Damaged: {damaged_count}")
print(f"Undamaged: {undamaged_count}")

# ==========================
# 💾 SAVE
# ==========================
cv2.imwrite("output.jpg", annotated)

# ==========================
# 👀 SHOW
# ==========================
cv2.imshow("Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Saved as output.jpg")
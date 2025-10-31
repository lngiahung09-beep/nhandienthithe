from ultralytics import YOLO
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

def draw_text_unicode(image, text, position, color=(255, 255, 255), size=28):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# 🧠 Model dò xác
fish_model_path = "/Users/billhhn/Downloads/chet dui.pt"
fish_model = YOLO(fish_model_path)
fish_model.to('mps')

# 🧍 Model dò pose
pose_model = YOLO("yolo11m-pose.pt")
pose_model.to('mps')

# 🎞️ Đường dẫn video
#video_path = "/Users/billhhn/Downloads/IMG_1171.MOV"
#cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở video.")
    exit()

# ⚙️ Lưu video output
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#out = cv2.VideoWriter("output_result.mp4", fourcc, 20.0,
#                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✳️ Dò xác
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fish_results = fish_model(gray_3ch, conf=0.3)

    # 🧍 Dò pose
    pose_results = pose_model(frame, conf=0.3)

    output_frame = frame.copy()

    if len(fish_results[0].boxes) > 0:
        for box in fish_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = fish_results[0].names[cls_id]
            conf = float(box.conf[0])

            # Vẽ khung và nhãn bằng Pillow (Unicode ok)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            output_frame = draw_text_unicode(
                output_frame,
                f"{label} ({conf:.2f})",
                (x1, y1 - 30),
                color=(255, 0, 0),
                size=28
            )
    else:
        # Không phát hiện → “Swim”
        output_frame = draw_text_unicode(output_frame, "Swim", (50, 50),
                                         color=(0, 255, 0), size=40)

    # 🧍 Vẽ keypoints pose (tắt box & nhãn để tránh lỗi font)
    pose_annotated = pose_results[0].plot(boxes=False, labels=False)

    # ⚡ Gộp
    combined = cv2.addWeighted(output_frame, 0.7, pose_annotated, 0.6, 0)

    out.write(combined)
    cv2.imshow("Nhan dien thi the", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
#print("🟢 Xong! File kết quả lưu tại: output_result.mp4")

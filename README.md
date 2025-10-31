🧠 Giới thiệu

Dự án được phát triển nhằm phát hiện và phân loại trạng thái con người dưới nước bằng mô hình YOLO11n-Pose, phục vụ công tác tìm kiếm và cứu hộ thi thể đuối nước.
Mô hình có khả năng nhận diện dáng người trong môi trường thiếu sáng, phân biệt giữa người đang bơi và đối tượng bất động có nguy cơ theo thời gian thực.

🎯 Mục tiêu

Ứng dụng thị giác máy tính để nhận diện người trong môi trường nước.

Phân loại hành vi: “bơi” 🟢 hoặc “đối tượng nguy cơ” 🔴 dựa trên pose.

Giảm thiểu rủi ro cho lực lượng cứu hộ, tăng tốc độ xác định vị trí nạn nhân.

Xây dựng hệ thống có thể chạy real-time trên máy tính hoặc thiết bị nhúng.

⚙️ Công nghệ sử dụng
Thành phần	Công nghệ
Framework AI	Ultralytics YOLOv11n-Pose
Ngôn ngữ	Python 3.10+
Xử lý ảnh	OpenCV, Pillow, NumPy
Môi trường huấn luyện	Google Colab (GPU Tesla T4)
Môi trường chạy thử	VSCode / Local Python
🧠 Quy trình huấn luyện

Thu thập dữ liệu từ video mô phỏng và tư liệu thực tế dưới nước.

Tách khung hình (frame) và gán nhãn thủ công với 2 lớp:

bơi → người đang hoạt động bình thường

đối tượng nguy cơ → người bất động, úp mặt hoặc không cử động

Huấn luyện YOLO11n-Pose trong Google Colab với cấu hình mặc định:

25 epoch

kích thước ảnh 640×640

batch = 16

learning rate = 0.01

Trích xuất trọng số mô hình (.pt) để sử dụng cho nhận diện real-time.

📊 Kết quả mô hình

mAP@50: 0.75

mAP@50-95: 0.63

Mô hình hội tụ ổn định sau 20 epoch, khả năng phân loại tốt hai trạng thái.

Hoạt động real-time (≈30 FPS) trên máy có GPU hoặc chip MPS (Mac).

💻 Triển khai

Mô hình được chạy local bằng VSCode/Python, sử dụng webcam hoặc video làm đầu vào.
Kết hợp hai mô hình:

chet_dui.pt → phát hiện thi thể

yolo11m-pose.pt → nhận diện khung xương (pose estimation)

Google Colab Notebook huấn luyện

🔮 Hướng phát triển

Mở rộng dataset với nhiều điều kiện ánh sáng và độ đục khác nhau.

Tối ưu tốc độ suy luận để triển khai trên Raspberry Pi / Jetson.

Kết hợp thêm cảm biến môi trường để tăng độ chính xác nhận diện.

👥 Nhóm thực hiện

Lê Nguyễn Gia Hưng – Lớp 11 Tin
Hoàng Hồ Anh Quang - Lớp 11 Tin

Trường THPT Chuyên Bảo Lộc

Lĩnh vực: Hệ thống nhúng

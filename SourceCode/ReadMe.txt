Khi test cần lưu ý:
- Máy nào yếu nên dùng colab chạy thời gian GPU vì CPU chạy không nổi
- Mỗi lần nhập xong 1 chữ, nhấn enter (không cần space). Chương trình sẽ xử lý đưa ra dự đoán, và in lại đoạn văn + new input.
- Trong trường hợp dự đoán đúng, ấn phím [`] sẽ tự nhận dự đoán là input.
- Muốn kết thúc nhập, ấn [//] sẽ break. Và đến phần thống kê.
- Trong bảng thống kê, cột đầu tiên là input mà người dùng nhập, cột 2 là từ dự đoán, cột 3 là đếm số lần đúng. 
- Input phải nằm trong vocabulary (không nằm trong bad_chars, không viết hoa).

Khi train cần lưu ý:
- bacth_size nên để lớn hơn 1000 (vì số ký tự rất lớn)
- epoch khoảng 50-100 ( sau khi thống kê, epochs=75 là đẹp) 
- model đã lưu là quá trình train 75 epoch với batch_size = 1024.

- file data_vtc_giao_duc.txt là tổng hợp từ thông tin giáo dục, Nguồn MiAI


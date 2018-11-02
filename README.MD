## Bài tập giữa kỳ môn tối ưu hóa

Họ Tên : Nguyễn Văn Phi
msv : 16021406

## 1,cài đặt 

* yêu cầu python3.6

        pip install -r requirements.txt

## 2, dạng bài toán

giải các bài toán dạng :

    tối ưu hàm( minimize ) f(x) = C.T * x
    
với các ràng buộc:
    
    A.T * x <= b
    x >= 0  

với A,c,b thuộc vào R. chương trình cho phép xác định trường hợp miền không đóng(không giải được)

## chạy thuật toán:

    python3 <file>.py 

* sử dụng chương trình simplex_op.py để tính nghiệm tối ưu và in ra bảng tableau

* sử dụng chương trình visualize.py(chỉ dùng với 2 biến thực) để vẽ miền giá trị của các bất phương trình,miền giá trị của hàm mục tiêu và điểm tối ưu lên hệ tọa độ decarters
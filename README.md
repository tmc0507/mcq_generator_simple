# MCQ Studio

Ứng dụng tạo bộ đề trắc nghiệm từ tài liệu và nhận xét bài làm sinh viên. Giao diện chạy bằng Streamlit; phần trích xuất tài liệu dùng C++, phần tạo câu hỏi và nhận xét dùng model local.

## Cấu trúc workspace

    input/
      documents/          Tài liệu PDF, DOC/DOCX, PPT/PPTX, CSV
      student_results/    Mỗi bài làm sinh viên là một file JSON
    output/
      questions.json      Bộ đề mới nhất
    models/
      modelv3.gguf        Model mặc định (4B)
      modelv2.gguf        Model lớn hơn (8B)

Ứng dụng tự tạo các thư mục cần thiết. File model, tài liệu mới tải lên, kết quả sinh viên và output không được đưa lên GitHub; hai tài liệu mẫu trong input/documents/ được giữ để thử chức năng. Các file tạm của CMake và quá trình chia tài liệu nằm trong build/ và work/.

## Chạy trên máy local

Cần Python 3, CMake, trình biên dịch C++, Poppler và unzip. Với file DOC/PPT cũ, cài thêm LibreOffice.

    sudo apt install build-essential cmake poppler-utils unzip python3-venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r scripts/requirements.txt
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    streamlit run mcq_streamlit.py

Đặt ít nhất models/modelv3.gguf hoặc models/modelv2.gguf vào workspace trước khi tạo đề. Giao diện mặc định chọn modelv3 khi có file. Model không nằm trong repository.

## Tạo đề và nhận xét

1. Trong chat, bấm dấu +, chọn tài liệu và nhập “Tạo 10 câu hỏi”. File tải lên được giữ trong input/documents/.
2. Sau khi tạo, toàn bộ JSON bộ đề xuất hiện ngay trong chat. App đồng thời ghi output/questions.json và có nút tải JSON.
3. Trang web làm bài ghi mỗi kết quả thành một file .json trong input/student_results/. Cũng có thể đính kèm file JSON bằng dấu +; app sẽ lưu bản đó vào cùng thư mục.
4. Hỏi “Nhận xét SV001” hoặc “Nhận xét buổi kiểm tra hôm nay”. App chấm theo bộ đề đang mở, lấy bản nộp mới nhất của mỗi sinh viên và chỉ tính file được ghi trong ngày khi hỏi về “hôm nay”.

Form kết quả:

    {
      "exam_id": "lay_tu_output_questions_json",
      "student_id": "SV001",
      "student_name": "Nguyễn Văn An",
      "answers": {
        "1": "", "2": "", "3": "", "4": "", "5": "",
        "6": "", "7": "", "8": "", "9": "", "10": ""
      }
    }

student_id và student_name là hai chuỗi bắt buộc, ghi riêng để tìm theo mã hoặc tên và quản lý nhiều lần nộp theo mã số. Khóa trong answers là ID câu hỏi; giá trị là A/B/C/D hoặc chuỗi rỗng. Form trên minh họa bộ đề 10 câu; khi dùng bộ đề khác, lấy đúng các ID từ output/questions.json. exam_id phải khớp bộ đề đang mở. Xem form trong examples/student_result.template.json và cặp đề/bài làm mẫu trong examples/world_model/.

Chế độ tạo nhanh giữ câu hỏi model sinh ra, chỉ loại câu trùng nguyên văn. Giảng viên cần rà soát đáp án trước khi giao đề. Có thể bật “Kiểm tra đáp án bằng AI” nếu cần kiểm tra sâu hơn.

## Chạy theo lệnh

    ./run.sh "input/documents/Slides CHAPTER - SVM.pdf" "models/modelv3.gguf" 10

Lệnh này build bộ đọc tài liệu, ghi work/chunks.json và tạo output/questions.json. Bộ đề tạo bằng lệnh cũng có exam_id để ghép với kết quả sinh viên.

## Chạy trên Kaggle

Bật GPU, tải workspace lên Kaggle, cài dependency và truyền đường dẫn model Transformers đã gắn vào notebook:

    bash kaggle/setup.sh
    ./kaggle/run_kaggle.sh "input/documents/World(1)(3).docx" /kaggle/input/my-model 10

Backend Kaggle dùng Transformers 4-bit; backend local dùng GGUF. Hai chế độ dùng chung prompt, schema, kiểm tra và định dạng đầu ra. Không cần mang file GGUF lên Kaggle.

## Kiểm thử

    python3 -m py_compile mcq_streamlit.py scripts/generate_mcq.py
    python3 test_generate_mcq.py
    python3 test_mcq_streamlit.py

Bộ đọc tài liệu có thể kiểm tra riêng bằng:

    ./build/doc_processor "input/documents/Slides CHAPTER - SVM.pdf" --output work/chunks.json

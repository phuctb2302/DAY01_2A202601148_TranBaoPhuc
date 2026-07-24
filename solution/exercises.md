# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi tăng `temperature` từ **0.0 → 1.8**, phản hồi của mô hình trở nên đa dạng và sáng tạo hơn. Ở mức **0.0**, câu trả lời ổn định và ít thay đổi; khoảng **0.7–1.2** vẫn mạch lạc nhưng có nhiều cách diễn đạt phong phú hơn. Đến khoảng **1.8**, phản hồi bắt đầu có xu hướng lan man, kém nhất quán hoặc đưa vào những chi tiết ít liên quan hơn.


### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Đối với **trợ lý soạn thảo hợp đồng pháp lý**, tôi sẽ chọn `temperature = 0.0` hoặc khoảng **0.1–0.2** để đảm bảo câu trả lời nhất quán, chính xác và hạn chế tính ngẫu nhiên. Đối với **trợ lý viết slogan quảng cáo**, tôi sẽ chọn khoảng **0.8–1.2** để mô hình tạo ra nhiều ý tưởng sáng tạo, đa dạng và hấp dẫn hơn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Với 20.000 người dùng, mỗi người gọi API 2 lần và mỗi lần sinh khoảng 500 output token, tổng lượng output là **20 triệu token/ngày**. Theo bảng giá, **GPT-4o** có chi phí khoảng **200 USD/ngày**, trong khi **GPT-4o-mini** chỉ khoảng **12 USD/ngày**.

Model lớn (GPT-4o) xứng đáng với chi phí trong các tác vụ yêu cầu độ chính xác và khả năng suy luận cao, chẳng hạn như phân tích tài liệu pháp lý hoặc hỗ trợ chuyên môn. Ngược lại, GPT-4o-mini là lựa chọn phù hợp cho chatbot hỏi đáp thông thường, hỗ trợ khách hàng hoặc các ứng dụng có lưu lượng lớn cần tối ưu chi phí.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Hai phản hồi có sự khác biệt rõ rệt. Với persona nhà thơ, câu trả lời sử dụng nhiều hình ảnh ví von, ngôn ngữ giàu cảm xúc, ít thuật ngữ kỹ thuật và dễ tiếp cận hơn. Với persona kỹ sư phần mềm senior, câu trả lời chính xác, có giải thích khái niệm chuyên môn và có thể kèm ví dụ code hoặc ứng dụng thực tế. Điều này cho thấy system prompt có thể điều khiển phong cách diễn đạt, mức độ chuyên môn, độ dài và cách trình bày của phản hồi, dù cùng trả lời một câu hỏi.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Khi so sánh, số token do tiktoken đếm thường không trùng với giá trị ước lượng số từ / 0.75. Trong thử nghiệm, hai kết quả chênh lệch khoảng 10–20% (tùy nội dung tiếng Việt). Nếu chỉ dùng công thức ước lượng để dự toán chi phí API cho tiếng Việt, chi phí thực tế có thể bị dự toán thiếu hoặc thừa, vì token không tương ứng trực tiếp với số từ; dấu câu, ký tự Unicode và cách mã hóa tiếng Việt đều ảnh hưởng đến số token thực tế.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Chatbot văn bản và đặc biệt là trợ lý giọng nói hưởng lợi nhiều từ streaming vì người dùng nhìn thấy hoặc nghe được phản hồi ngay khi mô hình bắt đầu sinh nội dung, giúp giảm cảm giác chờ đợi và tăng trải nghiệm tương tác. Trong đó, trợ lý giọng nói hưởng lợi nhiều nhất vì có thể phát âm từng phần của câu trả lời ngay khi nhận được. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm hầu như không cần streaming vì không có người dùng chờ trực tiếp; điều quan trọng hơn là độ chính xác, tính ổn định và hoàn thành toàn bộ tác vụ.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Khi API quá tải và nhiều client cùng retry, exponential backoff giúp các request được giãn cách theo thời gian thay vì liên tục gửi lại, từ đó giảm tải cho máy chủ và tăng khả năng request thành công ở lần thử sau. Nếu chỉ dùng delay cố định, rất nhiều client sẽ retry cùng lúc và tiếp tục gây quá tải. Kỹ thuật jitter bổ sung một khoảng thời gian ngẫu nhiên vào mỗi lần chờ để các client không retry đồng thời, giúp tránh hiện tượng thundering herd (nhiều client cùng gửi request một lúc), từ đó cải thiện độ ổn định của hệ thống.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *System prompt: "Bạn là trợ giảng AI thân thiện của khóa học AI. Hãy trả lời bằng tiếng Việt, ngắn gọn, chính xác, dễ hiểu và đưa ví dụ khi cần."

Hai phần quan trọng của prompt:
- "Trả lời bằng tiếng Việt": Nếu bỏ đi, trợ lý có thể trả lời bằng tiếng Anh hoặc ngôn ngữ khác tùy theo ngữ cảnh.
- "Ngắn gọn, chính xác, dễ hiểu và đưa ví dụ khi cần": Nếu bỏ đi, câu trả lời có thể dài dòng hơn, ít ví dụ hoặc trình bày khó hiểu hơn.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Bạn có thể điền ngắn gọn như sau:

### Câu 4.1 — Thiết kế persona

> System prompt: "Bạn là trợ giảng AI thân thiện, trả lời ngắn gọn bằng tiếng Việt, giải thích dễ hiểu và đưa ví dụ khi cần."
>
> * Nếu bỏ "trả lời ngắn gọn", câu trả lời sẽ dài và chi tiết hơn.
> * Nếu bỏ "giải thích dễ hiểu", câu trả lời có thể dùng nhiều thuật ngữ, khó hiểu hơn.

### Câu 4.2 — Hạn chế & cải thiện

> Nếu người dùng nhắc lại một chủ đề đã nói từ hơn 4 lượt trước, trợ lý có thể quên và trả lời sai ngữ cảnh. Có thể khắc phục bằng cách tóm tắt các lượt cũ" hoặc lưu lịch sử quan trọng thay vì chỉ giữ 4 lượt gần nhất.
*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)

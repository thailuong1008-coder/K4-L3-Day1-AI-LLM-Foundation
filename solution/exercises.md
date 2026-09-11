# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 đến 1.5, độ ngẫu nhiên và tính đa dạng trong lựa chọn token của mô hình tăng rõ rệt:
> - Ở **temperature 0.0**, phản hồi mang tính tất định (deterministic), tập trung vào dữ kiện phổ biến nhất (ví dụ: hang Sơn Đoòng là hang động tự nhiên lớn nhất thế giới) với câu từ cô đọng, gãy gọn và cấu trúc câu hầu như không đổi khi chạy lại nhiều lần.
> - Ở **temperature 0.5**, câu trả lời cân bằng tốt giữa tính chính xác và sự tự nhiên, diễn đạt mượt mà và lưu loát hơn.
> - Ở **temperature 1.0 và 1.5**, mô hình chọn các chủ đề phong phú hơn (như văn hóa, ẩm thực, địa danh ít phổ biến hơn), cách dùng từ giàu hình ảnh và sáng tạo hơn; tuy nhiên ở mức 1.5 câu văn bắt đầu có xu hướng hoa mỹ quá mức, câu từ dàn trải và tiềm ẩn nguy cơ xuất hiện thông tin ảo giác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đối với chatbot hỗ trợ khách hàng (Customer Support), mức temperature lý tưởng nên đặt trong khoảng **0.2 đến 0.4** (khuyến nghị mức **0.3**).
> **Lý do:**
> 1. **Độ chính xác và nhất quán (Consistency):** Khách hàng cần thông tin chính xác tuyệt đối về chính sách đổi trả, bảng giá, quy định bảo hành và quy trình khiếu nại. Temperature thấp giúp mô hình bám sát tài liệu/dữ liệu đối chiếu, hạn chế tối đa rủi ro "bịa đặt" (hallucination) có thể dẫn tới hậu quả pháp lý hoặc tổn hại uy tín doanh nghiệp.
> 2. **Văn phong thân thiện, tự nhiên:** Nếu đặt mức 0.0, chatbot sẽ phản hồi rất rập khuôn, cứng nhắc như máy móc; mức 0.2–0.3 vừa đủ để tạo sự mềm mại, lịch sự trong lời chào và giải thích mà vẫn giữ nguyên tính ổn định, tin cậy của thông tin.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> **1. Ước tính chi phí:**
> - Tổng số request mỗi ngày: $10.000 \text{ users} \times 3 = 30.000 \text{ requests/ngày}$.
> - Tổng lượng token đầu ra: $30.000 \times 350 = 10.500.000 \text{ tokens/ngày}$ (tương đương $10.500\text{k tokens}$).
> - Chi phí output GPT-4o ($0.010 / 1k token): $10.500 \times \$0.010 = \$105\text{/ngày}$ (khoảng **\$3.150 / tháng**).
> - Chi phí output GPT-4o-mini ($0.0006 / 1k token): $10.500 \times \$0.0006 = \$6,3\text{/ngày}$ (khoảng **\$189 / tháng**).
> $\rightarrow$ **GPT-4o đắt hơn GPT-4o-mini xấp xỉ 16,67 lần** ($0.010 / 0.0006$) đối với lượng token đầu ra của workload này.
> 
> **2. Trường hợp áp dụng phù hợp:**
> - **Nên dùng GPT-4o:** Các tác vụ phân tích logic phức tạp, tư vấn y tế/pháp lý, giải quyết khiếu nại mức độ nghiêm trọng hoặc hỗ trợ lập trình kiến trúc hệ thống — nơi mà độ chính xác và khả năng suy luận sâu là tối quan trọng, cái giá của một câu trả lời sai lệch lớn hơn rất nhiều so với chi phí gọi API.
> - **Nên dùng GPT-4o-mini:** Các tác vụ phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn, trả lời các câu hỏi thường gặp (FAQ đã có sẵn ngữ cảnh đối chiếu), hoặc các luồng tương tác với số lượng truy cập lớn cần tối ưu triệt để chi phí vận hành.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> **So sánh sự khác biệt giữa hai phản hồi:**
> - **Từ vựng & thuật ngữ:** Bản cho trẻ 8 tuổi dùng từ ngữ thuần Việt mộc mạc, gần gũi và tránh hoàn toàn thuật ngữ trừu tượng ("cuốn sổ ghi chép thần kỳ", "hộp khóa bí mật"); bản chuyên gia tài chính sử dụng dày đặc các thuật ngữ chuyên sâu (*distributed ledger technology - DLT*, *cryptographic hashing SHA-256*, *consensus mechanism Proof-of-Work/Proof-of-Stake*, *byzantine fault tolerance*, *immutability*).
> - **Ví dụ minh họa:** Bản giáo viên dùng hình ảnh ẩn dụ trực quan (cả lớp cùng giữ một bản sao của cuốn sổ ghi điểm chung nên không bạn nào tự ý sửa được điểm của mình); bản tài chính phân tích bài toán loại bỏ bên trung gian (disintermediation), tối ưu đối soát thanh toán xuyên biên giới và kiến trúc tài chính phi tập trung (DeFi).
> - **Độ dài & văn phong:** Bản giáo viên ngắn gọn, ấm áp, khích lệ; bản chuyên gia có kết cấu logic chặt chẽ, văn phong học thuật, trang trọng và đi sâu vào bản chất kỹ thuật.
> 
> $\rightarrow$ **Ảnh hưởng của System Prompt:** System prompt hoạt động như một "bộ định khung nhận thức" (cognitive framing), quy định vai trò, mức độ trừu tượng, đối tượng độc giả mục tiêu và giọng điệu phản hồi trước khi mô hình xử lý user prompt. Nó giúp định hướng trực tiếp không gian xác suất của mô hình mà không cần thay đổi hay can thiệp vào câu hỏi của người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> **1. Số liệu thực nghiệm:**
> - Khảo sát đoạn văn mẫu tiếng Việt gồm 106 từ:
>   - Ước lượng theo công thức tiếng Anh (`số từ / 0.75`): $106 / 0.75 \approx 141$ token.
>   - Đếm thực tế bằng thư viện `tiktoken` (bộ mã hóa `cl100k_base`): **227 token**.
>   - Độ chênh lệch: Số token thực tế **cao hơn ước lượng xấp xỉ 60,6%** (tỷ lệ thực tế đạt khoảng 2,14 token/từ thay vì 1,33 token/từ như tiếng Anh).
> 
> **2. Nguyên nhân tiếng Việt tốn nhiều token hơn tiếng Anh:**
> - Các thuật toán tokenizer hiện đại (như Byte-Pair Encoding - BPE) được xây dựng và tối ưu dựa trên kho ngữ liệu tiếng Anh khổng lồ, nơi hầu hết các từ vựng tiếng Anh nguyên khối đều có sẵn token ID đại diện trong bảng mã từ vựng (vocabulary).
> - Tiếng Việt là ngôn ngữ đơn lập có dấu thanh và dấu phụ (ă, â, đ, ê, ô, ơ, ư cùng các dấu sắc, huyền, hỏi, ngã, nặng). Trong chuẩn mã hóa UTF-8, các ký tự có dấu này chiếm từ 2 đến 3 byte. Khi tokenizer không chứa sẵn các từ ghép hoặc ký tự có dấu này trong từ điển, nó bắt buộc phải chia nhỏ từ thành nhiều ngữ tố (subwords) hoặc mã hóa theo từng byte đơn lẻ. Do đó, một từ tiếng Việt thông thường dễ bị phân tách thành 2–3 token, làm tăng mạnh số token tiêu tốn cũng như chi phí sử dụng API.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> **Streaming** quan trọng nhất trong các ứng dụng đối thoại trực tiếp với con người (giao diện Chatbot, CLI, trợ lý viết lách, giải toán), nơi câu trả lời dài có thể mất 10–30 giây để sinh xong hoàn toàn; việc hiển thị ngay từng từ giúp giảm thời gian chờ đợi phản hồi đầu tiên (Time to First Token - TTFT) xuống dưới 1 giây, tạo cảm giác hệ thống phản xạ tức thì và giữ chân người dùng. Ngược lại, **non-streaming** phù hợp hơn trong các tác vụ ngầm (background jobs, ETL pipeline, chấm điểm tự động), các bài toán cần trích xuất dữ liệu có cấu trúc (Structured Outputs / JSON mode để `json.loads` toàn vẹn), gọi công cụ (Function/Tool Calling), hoặc khi cần chạy qua bộ lọc kiểm duyệt an toàn (content moderation filter) toàn bộ văn bản trước khi hiển thị cho người xem.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> **1. Lợi thế của Exponential Backoff:**
> Khi hệ thống API gặp sự cố quá tải (mã lỗi 429 Too Many Requests hoặc 503 Service Unavailable), máy chủ cần thời gian để xử lý hết hàng đợi hiện tại và giải phóng tài nguyên. Exponential backoff kéo giãn thời gian chờ theo hàm mũ ($base\_delay \times 2^{attempt}$: 0.1s, 0.2s, 0.4s, 0.8s...), giúp giảm theo cấp số nhân mật độ request gửi tới máy chủ, tạo khoảng lặng cần thiết cho hệ thống tự phục hồi.
> 
> **2. Hậu quả khi dùng delay cố định (Thundering Herd Problem):**
> Nếu hàng nghìn client cùng gặp lỗi tại thời điểm $T$ và đều đợi đúng 1 giây để retry, thì đúng tại thời điểm $T + 1s$, toàn bộ hàng nghìn request đó sẽ đồng loạt dội ngược trở lại máy chủ như một đợt "sóng thần" (retry storm). Hiện tượng này (Thundering Herd) khiến máy chủ vừa chớm khởi động lại đã ngay lập tức bị sập tiếp, dẫn đến tình trạng tê liệt kéo dài và làm trầm trọng thêm sự cố nghẽn mạng.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> **1. System Prompt lựa chọn:**
> `"Bạn là Trợ giảng AI thân thiện hỗ trợ sinh viên học lập trình Python. Hãy trả lời ngắn gọn, súc tích bằng tiếng Việt, luôn kèm theo một ví dụ code minh họa đơn giản và giải thích từng bước rõ ràng."`
> 
> **2. Phân tích lựa chọn từ ngữ quan trọng:**
> - *"Trả lời ngắn gọn, súc tích"*: Cực kỳ thiết yếu trong môi trường dòng lệnh (CLI Terminal), nơi không gian hiển thị có hạn và việc cuộn trang liên tục gây mất tập trung. Giới hạn này giúp câu trả lời đi thẳng vào trọng tâm, đồng thời tiết kiệm đáng kể token đầu ra và giảm độ trễ phản hồi.
> - *"Bằng tiếng Việt" kèm "ví dụ code minh họa đơn giản"*: Việc chỉ định rõ tiếng Việt giúp ngăn mô hình tự động chuyển ngữ sang tiếng Anh khi gặp các từ khóa kỹ thuật phổ biến. Yêu cầu kèm ví dụ code ngắn giúp sinh viên trực quan hóa ngay lý thuyết vào thực hành thay vì chỉ đọc giải thích trừu tượng.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **1. Hạn chế lớn nhất:**
> Trợ lý hiện tại hoàn toàn thiếu **bộ nhớ dài hạn (Long-term Memory & Persistence)** và bị giới hạn cứng ngữ cảnh trong 3 lượt hội thoại gần nhất (`history[-6:]`) lưu tạm thời trong RAM. Khi người dùng tắt terminal hoặc cuộc trò chuyện kéo dài qua lượt thứ 4, các thông tin ngữ cảnh quan trọng ban đầu (như tên sinh viên, bài toán đang giải, lỗi gặp phải ở các bước trước) sẽ bị mất hoàn toàn.
> 
> **2. Đề xuất cải thiện & Cách triển khai:**
> **Giải pháp: Cơ chế Lưu trữ phiên & Tóm tắt ngữ cảnh động (Persistent Session & Context Summarization)**
> - *Lưu trữ (Persistence):* Lưu toàn bộ lịch sử hội thoại vào file SQLite cục bộ hoặc file JSON theo `session_id`. Khi người dùng mở lại CLI, chương trình sẽ nạp lại phiên làm việc trước đó thay vì khởi động danh sách rỗng.
> - *Tóm tắt ngữ cảnh (Summarization Buffer):* Khi lịch sử vượt quá 6 messages, thay vì cắt bỏ thẳng tay các tin nhắn cũ, hệ thống sẽ kích hoạt một tiến trình ngầm gọi model nhẹ (`gemini-3.5-flash-lite` hoặc `gpt-4o-mini`) để cô đọng nội dung các lượt cũ thành một đoạn tóm tắt súc tích (ví dụ: *"Người dùng đang gỡ lỗi hàm divide_by_zero trong file test.py"*). Đoạn tóm tắt này được chèn vào làm ngữ cảnh nền tảng ngay sau System Prompt, giúp trợ lý duy trì sự hiểu biết xuyên suốt mà không làm bùng nổ số lượng token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026

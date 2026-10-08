# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Việt Hoàng Hải  **MSSV**: 2A202602967  **Ngày**: 08/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4, 14,56 GiB VRAM

> Bản nháp đang hoàn thiện. NB1 đã chạy trên CPU; các số đo NB2–NB5 sẽ được điền từ
> artefact thật sau khi có GPU. Không dùng số ước lượng thay cho kết quả thí nghiệm.

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 theo tier T4; p95 đo được 98, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch theo mặc định; số step sẽ lấy từ `results/runs.csv` sau NB3 |

**Template có giữ khối `<think>` không?** Có. `template_check.json` ghi
`open_tag_present=true`, `body_present=true`. Corpus mặc định chỉ có đáp án JSON,
không chứa reasoning trace để fine-tune.

**Lựa chọn thí nghiệm.** Model 4B là cấu hình mặc định cho T4 16 GB trong lab; tôi
giữ cùng model cho baseline và LoRA để phép so sánh công bằng. Corpus CSKH tiếng Việt
có nhãn 4 trường, giúp chấm tự động theo từng trường và kiểm tra định dạng JSON.
Tập đánh giá gồm 50 ticket target và 15 câu regression; tôi sẽ đo baseline (a), (b)
trước khi train. `max_length=1024` hiện là trần của tier. Độ dài lớn nhất đo được chỉ
101 token, và T4 dùng batch 1 nên không phát sinh padding giữa các mẫu; tôi sẽ ghi rõ
sự khác biệt với mức 256 do p95 gợi ý thay vì coi 1024 là con số đo được.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token ở mẫu kiểm tra) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3387.2 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1027.6 |
| (c) LoRA fine-tune | | | | |

**(b) có thật sự mạnh hơn (a) không?** Có: target tăng từ 0 lên 0.765,
format từ 0 lên 1.000. Tôi giữ nguyên `OPTIMIZED_PROMPT` của repo (SHA
`719e74d3b6232053`). NB2 hoàn tất và đóng băng trước khi NB3 bắt đầu trên Colab.
Tập đánh giá được dùng đầy đủ: 50 ticket target và 15 câu regression; không đặt `EVAL_LIMIT`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | | | | | | |
| `attn_only` | q,v | *(matched)* | | | | | | |
| `wrong_lr` | text-linear | 16 | | | | | | |
| `qlora` | text-linear | 16 | | | | | | |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `<PASSED | FAILED>`
`target Δ = <+0.xxx>` · `regression Δ = <+0.xxx>` · `valid_trace_rate = <0.xx>`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | | | | | ✅ FT thắng |
| 2 | | | | | ✅ FT thắng |
| 3 | | | | | ❌ **FT thua** |
| 4 | | | | | ❌ **FT thua** |
| 5 | | | | | |

Có mẫu chung nào ở các ca FT thua không?

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

**Ba điều tôi học được** (cụ thể, không generic):
1.
2.
3.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

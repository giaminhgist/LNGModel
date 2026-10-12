# Missing artifacts — presentation figure suite (EMS-Projects)

Danh sách những gì chưa có / không thể có trong repo hiện tại, lý do và lệnh
bổ sung nếu cần. Các scripts figure hiện tại đã chạy hết trên artifacts sẵn có;
không có chỗ nào bịa dữ liệu để lấp chỗ trống.

## 1. λ_norm — đã bị loại khỏi protocol trước các run 10-seed

- **Lý do**: Suite của repo này không có `lambda_norm` (đã bỏ trước khi chạy
  10 seeds; xem `EXPERIMENTS_RESULTS.md`). Vì vậy Figure_4 không thể tái lập
  đúng 2 panel so sánh λ=0 vs λ=0.1 của reference suite.
- **Cách xử lý trong bộ này**: các panel so sánh λ bị loại khỏi Figure_4.
  Panel còn lại duy nhất theo hướng đó là phân phối train-HC ‖z−μ‖ (kiểm tra
  concentration); *effective rank* (participation ratio) của train-HC
  covariance vẫn được tính và lưu ở `tables/T05.01_norm_diagnostics.csv`
  như chẩn đoán không vẽ.
- **Lệnh bổ sung** (nếu muốn đúng nguyên mẫu): train thêm một ablation
  learned + λ_norm=0.1 (cần sửa `src/models/proposal.py` + `src/training/trainer.py`, chưa có
  sẵn trong code) rồi export latent như hiện tại.

## 2. SHAP / Integrated Gradients — chưa chạy

- **Lý do**: `shap` không có trong environment; permutation importance (đã làm,
  Figure_5) đủ trả lời câu hỏi feature-level và family-level mà không cần
  assumption về attributions đi qua từng tầng.
- **Lệnh bổ sung**: `uv add shap` rồi viết script SHAP cho head (ghi rõ
  background từ training data, target = logit).

## 3. Không có metadata lâm sàng

- **Thiếu**: tuổi, giới, PANSS/clinical scores của từng subject. Repo không
  commit các trường này; paper nói hai nhóm matched nhưng không có bảng số
  theo subject.
- **Ảnh hưởng**: Không vẽ/kiểm soát confound nhân khẩu (giới hạn của
  Figure_1/Figure_2); mọi kết luận HC/SZ là association, không causal.

## 4. UMAP / probing bổ sung — cố ý không làm

- Figure_4 dùng PCA (fit trên training-HC reference) — đủ cho câu hỏi
  "SZ có lệch khỏi HC norm trong latent space không". UMAP chỉ bổ sung nếu
  PCA không tách được; ở đây PC1 đã tách rõ (p = 2.4e−13) nên không cần.
- OOF embeddings ghép qua các model khác nhau không phải common space nên
  không vẽ chung PCA từ embeddings của nhiều model.

## 5. Ghi chú về port từ reference suite

- Figures 1–3: tables T01.*/T02.* đã được đối chiếu tự động với
  `EMS-Project/presentation/tables/` — khớp chính xác (max diff 0 hoặc 1e−14).
- Figures 4–5: model của repo này (deepset main thay cho attention-norm của
  reference) nên các số và phần kết luận trong `figure_index.md` được viết
  lại theo số thực của run này, không copy từ reference.

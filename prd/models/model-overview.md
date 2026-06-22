# ARCHITECTURE & EXPERIMENT DESIGN

Proyek ini membandingkan arsitektur pada dua sumbu — **algoritma** (PPO vs DQN) dan **representasi state** (12-bit vector vs spatiotemporal tensor). Spatiotemporal saat ini menggunakan **4-channel honest layout** (Wall, Decaying body, Static food, Dynamic food momentum) — Ch4-Ch7 dari layout v2 lama (head dir, food dir, danger, broadcast snake length) sudah dihapus agar perbandingan mengukur algoritma, bukan _cheat-sheet_. Keempat kombinasinya diimplementasikan di `game/model/`:

| File                               | Algo | Obs Type                 | Extractor                                   | Head           |
| ---------------------------------- | ---- | ------------------------ | ------------------------------------------- | -------------- |
| `game/model/ppo_spatiotemporal.py` | PPO  | spatiotemporal (4×20×20) | `SpatiotemporalExtractor` (CNN + Attention) | actor + critic |
| `game/model/ppo_12bit.py`          | PPO  | 12bit (12,)              | `DQN12BitExtractor` (MLP 12→64→64→64)       | actor + critic |
| `game/model/dqn_spatiotemporal.py` | DQN  | spatiotemporal (4×20×20) | `SpatiotemporalExtractor` (CNN + Attention) | Q-head         |
| `game/model/dqn_12bit.py`          | DQN  | 12bit (12,)              | `DQN12BitExtractor` (MLP 12→64→64→64)       | Q-head         |

Kedua `SpatiototemporalExtractor` (PPO & DQN) **identik secara byte-per-byte** karena di-import ulang oleh `dqn_spatiotemporal.py`. Begitu juga `DQN12BitExtractor` dipakai ulang oleh `ppo_12bit.py`. Pengulangan yang disengaja ini menjamin bahwa perbandingan algoritma tidak tercemar oleh perbedaan encoder — satu-satunya variabel antar algoritma adalah algoritmanya sendiri.

## 1. Proposed Model: Spatiotemporal PPO

Algoritma PPO (_On-Policy_) dipilih untuk menghindari _memory leak_ dari _Replay Buffer_ DQN saat memproses tensor matriks masif.

- **Spatial Extractor (compact):** 2 layer **stride-2 Conv2D + ReLU**. Input 20×20×4 (honest layout — Ch0 Wall, Ch1 Decaying body, Ch2 Static food, Ch3 Dynamic food) didownsample 20×20 → 10×10 → 5×5. Setiap token output punya **7×7 receptive field** di input space (vs. 5×5 pada varian stride-1 lama). Trade-off: lebih besar RF per token, 16× lebih sedikit token untuk attention.
- **Temporal Attention (compact):** 1 layer Transformer Encoder (_Multi-Head Attention_) di atas **25 token spasial** (5×5) + 1 learnable `[CLS]` token. Biaya attention: 26² ≈ 676 scores per sample — sekitar 240× lebih murah dari varian 400-token lama (401² ≈ 160k scores), tanpa kehilangan kemampuan penalaran global lintas papan.

### Mengapa compact (strided CNN + sedikit token)?

Varian lama (stride-1, 400 token) membengkak karena setiap sel grid menjadi token attention terpisah. Mayoritas token itu tetangga spatialnya yang sudah bisa di-capture CNN lokal; Transformer jadi belajar ulang hubungan adjacent yang sebenarnya _spatial bias_-nya sudah hilang. Stride-2 CNN memaksa representasi spasial untuk **dikompresi secara hierarkis** dulu — tiap token 5×5 berisi informasi tentang region 4×4 input dengan RF 7×7 — sehingga Transformer cukup fokus pada **relasi global** (food position vs body trap vs open area) yang betul-betul perlu penalaran attention.

## 2. Baseline Model: DQN 12-bit (Paper Replication)

Algoritma DQN menggunakan arsitektur _Dense Layer_ (MLP) murni. Menerima input data 1D (_flattened_) sebesar 12-bit sesuai literatur.

## 3. PPO 12-bit (algoritma modern, state baseline)

PPO + MLP 12-bit memvalidasi apakah PPO sendirian sudah cukup untuk menutup gap dengan baseline paper. Input mengikuti spec 12-bit yang sama dengan DQN paper, head-nya adalah actor + critic terpisah (lihat `game/model/ppo_12bit.py`).

## 4. DQN Spatiotemporal (algoritma baseline, state modern)

DQN + CNN+Attention (compact, sama dengan PPO) memvalidasi apakah representasi spasial modern membantu _off-policy_ learning. Features extractor di-reuse dari PPO (lihat §1), hanya head-nya diganti dari actor/critic ke Q-head. Flag `use_attention=False` mengaktifkan ablation CNN-only (5×5 flatten → MLP, tanpa Transformer) untuk percobaan isolated — default config sekarang `use_attention=true`; ablation dilakukan dengan override per-run dari TUI launcher.

## Rancangan Eksperimen

1. **Curriculum Environment Study:** Melatih PPO dan DQN secara sekuensial dari Level 1 hingga 5 (_Continuous Learning_). Evaluasi dilakukan melalui kurva rata-rata hadiah (_ep_rew_mean_) dan snake length (`rollout/snake_length_mean`) di TensorBoard.
2. **Architecture Ablation Study:** Menghapus/menonaktifkan _Transformer layer_ (menjadi CNN-only via `use_attention=False`). Dilatih pada Level 5 untuk memvalidasi peran modul _Attention_ dalam resolusi intersep dinamis. Dapat dijalankan pada PPO maupun DQN.
3. **PPO vs DQN Cross-Comparison:** Karena keempat varian tersedia, eksperimen dapat dipisah:
   - **Algoritma effect** (architecture dikontrol): PPO-sptmp vs DQN-sptmp dan PPO-12bit vs DQN-12bit.
   - **Architecture effect** (algoritma dikontrol): PPO-sptmp vs PPO-12bit dan DQN-sptmp vs DQN-12bit.
4. **PPO Hyperparameter Tuning:** Menguji parameter laju pembelajaran konsevatif ($1 \times 10^{-4}$) melawan laju agresif ($5 \times 10^{-4}$) untuk melihat stabilitas _Policy Entropy_.

## Konfigurasi Hyperparameter via JSON

Hyperparameter default tersimpan di [`game/model/configs/ppo_config.json`](../../game/model/configs/ppo_config.json) dan [`game/model/configs/dqn_config.json`](../../game/model/configs/dqn_config.json). Trainer (`PPOTrainingConfig.from_json_dict()`, `DQNTrainingConfig.from_json_dict()`) memuat file JSON sebagai basis, lalu TUI launcher menambahkan override per-run (level, obs_type, total_timesteps, learning_rate, dll.) lewat kwargs. Tidak ada hyperparameter yang di-hardcode di trainer — semuanya single source of truth di JSON. Lihat `prd/models/train.md § Konfigurasi Trainer` untuk tabel field lengkap dan default value.

## Konfigurasi Reward

Sistem reward ini dirancang untuk memberikan sinyal belajar yang kaya (_reward shaping_) dengan memanfaatkan jarak Euclidean terhadap target terdekat.

### Reward Events

| Event                          | Reward | Deskripsi                       |
| ------------------------------ | ------ | ------------------------------- |
| Bergerak **mendekati** makanan | +1.0   | Menghargai efisien pendekatan   |
| Bergerak **menjauhi** makanan  | -0.5   | Menghukum gerakan membingungkan |
| Memakan makanan statis         | +10    | Reward utama untuk menang       |
| Memakan makanan dinamis        | +8     | Reward utama untuk menang       |
| Tabrakan (dinding/tubuh)       | -10    | Punishment fatal                |
| Time penalty                   | -0.001 | Tekanan minimal untuk efisiensi |

> **Catatan:** nilai di atas adalah _baseline_ PRD. Nilai aktual yang dipakai runtime dibaca dari [`game/env/config.json`](../../game/env/config.json) (lihat `REWARD_APPROACH`, `REWARD_AWAY`, `REWARD_EAT_STATIC`, `REWARD_EAT_DYNAMIC`, `REWARD_COLLISION`, `REWARD_TIME`, plus _milestone bonus_ dan _stagnation penalty_ opsional — lihat `prd/game/environment.md`).

### Implementasi Pseudo-code

```python
def calculate_reward(old_head, new_head, food_eaten, collision):
    reward = 0

    # 1. Distance-based reward shaping
    old_dist = euclidean(old_head, nearest_food)
    new_dist = euclidean(new_head, nearest_food)
    if new_dist < old_dist:
        reward += 1.0   # Moved closer
    else:
        reward -= 0.5   # Moved away

    # 2. Food eaten
    if food_eaten == "static":
        reward += 10
    elif food_eaten == "dynamic":
        reward += 8

    # 3. Collision penalty
    if collision:
        reward -= 10

    # 4. Time efficiency
    reward -= 0.001

    return reward
```

### Desain Rationale

- **Distance reward (+1.0/-0.5):** Memberikan sinyal kontinu setiap step, mempercepat konvergensi dibanding reward sparce.
- **Food reward (+10):** Skala besar untuk memperkuat goal utama.
- **Collision penalty (-10):** Cukup besar untuk diajarkan avoidance, tapi tidak overpower dibanding food reward.
- **Time penalty (-0.001):** Minimal agar tidak terlalu menghukum gerakan aman di late-game.
- **Balancing:** Makanan Dinamis diberi reward sedikit lebih rendah untuk mencerminkan tantangan tambahan, mendorong strategi yang lebih adaptif. karena target dinamis mungkin membawa risiko lebih tinggi untuk mengejar.

### PPO Spatiotemporal Architecture Diagram

```
[Input: Spatiotemporal Tensor — honest layout]
     Shape: (4, 20, 20)
  Ch0 Wall          (1.0 on the four border rows/cols)
  Ch1 Decaying body (head=1.0, decays linearly to tail;
                     extent also implicitly encodes length)
  Ch2 Static food   (1.0 at each static food cell)
  Ch3 Dynamic food  (1.0 at current cell, 0.5 at previous —
                     momentum marker critical on levels 3-4)
             |
             v
+----------------------------------------+
|   COMPACT SPATIAL EXTRACTOR (CNN)      |
|----------------------------------------|
| - Conv2D(stride=2) Layer (ReLU)        |  20×20 → 10×10
| - Conv2D(stride=2) Layer (ReLU)        |  10×10 →  5×5
+----------------------------------------+
# Setiap token output 5×5 punya receptive field 7×7 di input.
# Bandingkan dengan varian lama (stride=1, 20×20 output): RF 5×5
# per token tapi 400 token (16× lebih banyak) dibutuhkan attention.
#
# Layout v2 sebelumnya (8 ch dengan head dir, food dir, danger,
# snake-length broadcast) sudah dihapus — itu semua adalah _heuristic
# crutches_ yang membuat jaringan membaca jawabannya secara langsung
# alih-alih mempelajarinya dari geometri mentah. Ch3 (dynamic food
# per-cell map dengan 1.0 current + 0.5 previous) tetap dipertahankan
# karena penting untuk level 3-4 di mana makanan dinamis adalah
# satu-satunya target — tanpa marker momentum ini, agen tidak punya
# cara mengetahui ke arah mana makanan bergerak.
             |
      [25 tokens × 32-dim]   ← (B, 25, 32) setelah flatten spatial
             |
             v
+----------------------------------------+
| TEMPORAL ATTENTION MODULE              |
|----------------------------------------|
| - Linear(32 → 64) cell embedding       |
| - Learnable [CLS] token prepended      |  → (B, 26, 64)
| - Transformer Encoder (1 layer)        |
|   • Multi-Head Attention (4 heads)     |  26² = 676 attention scores
|   • Feed-Forward Network (256-dim)     |
| - Linear(64 → 64) output projection    |
+----------------------------------------+
             |
      [Context Vector]  (B, 64)
             |
      +------+------+
      |             |
      v             v
+-----------+ +-----------+
| ACTOR     | | CRITIC    |
| HEAD      | | HEAD      |
| (Linear)  | | (Linear)  |
+-----------+ +-----------+
      |             |
      v             v
  [Action]       [Value]
 Probabilities   Estimate
  (4 logits)     (Scalar)
```

### PPO 12-bit Architecture Diagram

```
[Input: 12-bit Vector]
     Shape: (12,)
 Bits 0-3 : obstacle (wall/body) di [UP, RIGHT, DOWN, LEFT]
 Bits 4-7 : body proximity (1 langkah ke depan)
 Bits 8-11: relative food direction (signs)
             |
             v
+-----------------------------+
|  DENSE NETWORK (MLP BASE)   |
|-----------------------------|
| - Linear(12 → 64) + ReLU    |
| - Linear(64 → 64) + ReLU    |
| - Linear(64 → 64) + ReLU    |
+-----------------------------+
             |
      [Context Vector]  (B, 64)
             |
      +------+------+
      |             |
      v             v
+-----------+ +-----------+
| ACTOR MLP | | CRITIC MLP|
| (64→64+ReLU)  (64→64+ReLU)|
+-----------+ +-----------+
      |             |
      v             v
  [Action]       [Value]
  (4 logits)     (Scalar)
```

MLP features extractor di-reuse dari `dqn_12bit.py` (3 layer Dense, identik dengan paper baseline) supaya perbandingan PPO↔DQN pada state 12-bit benar-benar apples-to-apples.

### DQN Spatiotemporal Architecture Diagram

```
[Input: Spatiotemporal Tensor — honest layout]
     Shape: (4, 20, 20)
             |
             v
+----------------------------------------+
|   COMPACT SPATIAL EXTRACTOR (CNN)      |  (Sama dengan PPO sptmp)
|----------------------------------------|
| - Conv2D(stride=2) Layer (ReLU)        |  20×20 → 10×10
| - Conv2D(stride=2) Layer (ReLU)        |  10×10 →  5×5
+----------------------------------------+
             |
      [25 tokens × 32-dim]   ← (B, 25, 32)
             |
             v
+----------------------------------------+
| TEMPORAL ATTENTION MODULE              |  (Sama dengan PPO sptmp)
|----------------------------------------|
| - Linear(32 → 64) cell embedding       |
| - Learnable [CLS] token prepended      |
| - Transformer Encoder (1 layer, 8 hd)  |
| - Linear(64 → 64) output projection    |
+----------------------------------------+
             |
      [Context Vector]  (B, 64)
             |
             v
+-----------------------------+
|        Q-VALUE HEAD         |
|-----------------------------|
| - Linear(64 → 4)            |
+-----------------------------+
             |
             v
      [Action Q-Values]
(Q_Up, Q_Right, Q_Down, Q_Left)
             |
             v
        [argmax(Q)]
       Greedy Action
```

> `use_attention=False` (toggle di [`dqn_config.json`](../../game/model/configs/dqn_config.json) atau lewat trainer) menonaktifkan blok Temporal Attention dan menggantinya dengan flatten+Linear langsung dari output 5×5 CNN — itulah _architecture ablation_ yang bisa dijalankan identik pada PPO maupun DQN. **Default config sekarang `use_attention=true`**; ablation dilakukan via override per-run dari TUI launcher.

### DQN 12-bit Architecture Diagram

```
[Input: 12-bit Vector]
     Shape: (12,)
 (Obstacles, Food Direction)
             |
             v
+-----------------------------+
|  DENSE NETWORK (MLP BASE)   |
|-----------------------------|
| - Linear/Dense Layer (ReLU) |
| - Linear/Dense Layer (ReLU) |
| - Linear/Dense Layer (ReLU) |
+-----------------------------+
             |
      [Hidden Features]
             |
             v
+-----------------------------+
|        Q-VALUE HEAD         |
|-----------------------------|
| - Linear/Dense Layer        |
+-----------------------------+
             |
             v
      [Action Q-Values]
(Q_Up, Q_Right, Q_Down, Q_Left)
             |
             v
        [argmax(Q)]
       Greedy Action
```

# ARCHITECTURE & EXPERIMENT DESIGN

Proyek ini membandingkan arsitektur pada dua sumbu — **algoritma** (PPO vs DQN) dan **representasi state** (12-bit vector vs spatiotemporal tensor). Keempat kombinasinya diimplementasikan di `game/model/`:

| File                               | Algo | Obs Type       | Extractor                                   | Head           |
| ---------------------------------- | ---- | -------------- | ------------------------------------------- | -------------- |
| `game/model/ppo_spatiotemporal.py` | PPO  | spatiotemporal | `SpatiotemporalExtractor` (CNN + Attention) | actor + critic |
| `game/model/ppo_12bit.py`          | PPO  | 12bit          | `DQN12BitExtractor` (MLP 12→64→64→64)       | actor + critic |
| `game/model/dqn_spatiotemporal.py` | DQN  | spatiotemporal | `SpatiotemporalExtractor` (CNN + Attention) | Q-head         |
| `game/model/dqn_12bit.py`          | DQN  | 12bit          | `DQN12BitExtractor` (MLP 12→64→64→64)       | Q-head         |

Kedua `SpatiototemporalExtractor` (PPO & DQN) **identik secara byte-per-byte** karena di-import ulang oleh `dqn_spatiotemporal.py`. Begitu juga `DQN12BitExtractor` dipakai ulang oleh `ppo_12bit.py`. Pengulangan yang disengaja ini menjamin bahwa perbandingan algoritma tidak tercemar oleh perbedaan encoder — satu-satunya variabel antar algoritma adalah algoritmanya sendiri.

## 1. Proposed Model: Spatiotemporal PPO

Algoritma PPO (_On-Policy_) dipilih untuk menghindari _memory leak_ dari _Replay Buffer_ DQN saat memproses tensor matriks masif.

- **Spatial Extractor:** 2 layer Conv2D + ReLU. Mengekstrak relasi geometrik lokal dari tensor 20×20×8 (atau 20×20×4 untuk v1 legacy).
- **Temporal Attention:** 1 layer Transformer Encoder (_Multi-Head Attention_) di atas 400 sel spasial + 1 learnable `[CLS]` token. Menghitung prioritas lintas-trajektori target dinamis.

## 2. Baseline Model: DQN 12-bit (Paper Replication)

Algoritma DQN menggunakan arsitektur _Dense Layer_ (MLP) murni. Menerima input data 1D (_flattened_) sebesar 12-bit sesuai literatur.

## 3. PPO 12-bit (algoritma modern, state baseline)

PPO + MLP 12-bit memvalidasi apakah PPO sendirian sudah cukup untuk menutup gap dengan baseline paper. Input mengikuti spec 12-bit yang sama dengan DQN paper, head-nya adalah actor + critic terpisah (lihat `game/model/ppo_12bit.py`).

## 4. DQN Spatiotemporal (algoritma baseline, state modern)

DQN + CNN+Attention memvalidasi apakah representasi spasial modern membantu _off-policy_ learning. Features extractor di-reuse dari PPO (lihat §1), hanya head-nya diganti dari actor/critic ke Q-head. Flag `use_attention=False` mengaktifkan ablation CNN-only untuk percobaan isolated.

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
[Input: Spatiotemporal Tensor v2]
     Shape: (8, 20, 20)
  Ch0 Wall          | Ch4  Head dir       (1.0 at cell 1 step in current
  Ch1 Decaying body |                        direction)
  Ch2 Static food   | Ch5  Food direction (1.0 at the 4 cells around the
  Ch3 Dynamic food       head in any direction where SOME food exists;
        (1.0 current,    mirrors 12-bit obs's Bits 8-11)
         0.5 previous) | Ch6  Relative danger(1.0 at the 3 cells STRAIGHT /
                                          LEFT / RIGHT of head — relative
                                          to current heading — if wall or
                                          body; "behind" omitted since 180°
                                          reversal is impossible)
                    | Ch7  Snake length   (broadcast len/400 everywhere)
             |
             v
+-----------------------------+
|   SPATIAL EXTRACTOR (CNN)   |
|-----------------------------|
| - Conv2D Layer (ReLU)       |
| - Conv2D Layer (ReLU)       |
| - Flatten()                 |
+-----------------------------+

# Catatan: legacy 4-channel v1 obs (Wall, Body, Static, Dynamic) masih
# didukung via `obs_type="spatiotemporal_legacy"` agar model lama tidak
# rusak. Layout v2 menambahkan head direction, food direction, relative
# danger (3 sel relative to heading), dan snake length di atas 4 ch v1.
# Ch3 (dynamic food per-cell map dengan 1.0 current + 0.5 previous) adalah
# sinyal penting untuk level 3-4 yang makanan dinamis adalah satu-satunya
# target — tanpa channel ini, agen hanya punya directional signal di Ch5.
             |
      [Feature Vector]
             |
             v
+-----------------------------+
| TEMPORAL ATTENTION MODULE   |
|-----------------------------|
| - Transformer Encoder       |
| - Multi-Head Attention      |
| - Feed-Forward Network      |
+-----------------------------+
             |
      [Context Vector]
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
[Input: Spatiotemporal Tensor v2]
     Shape: (8, 20, 20)
             |
             v
+-----------------------------+
|   SPATIAL EXTRACTOR (CNN)   |  (Sama dengan PPO sptmp)
|-----------------------------|
| - Conv2D Layer (ReLU)       |
| - Conv2D Layer (ReLU)       |
+-----------------------------+
             |
             v
+-----------------------------+
| TEMPORAL ATTENTION MODULE   |  (Sama dengan PPO sptmp)
|-----------------------------|
| - Transformer Encoder       |
| - Multi-Head Attention      |
+-----------------------------+
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

> `use_attention=False` (toggle di [`dqn_config.json`](../../game/model/configs/dqn_config.json) atau lewat trainer) menonaktifkan blok Temporal Attention dan menggantinya dengan flatten+Linear langsung dari output CNN — itulah _architecture ablation_ yang bisa dijalankan identik pada PPO maupun DQN.

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

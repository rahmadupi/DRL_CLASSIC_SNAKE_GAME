# ARCHITECTURE & EXPERIMENT DESIGN

Proyek ini membandingkan dua arsitektur algoritmik yang dipisahkan di direktori `game/models/`.

## 1. Proposed Model: Spatiotemporal PPO

Algoritma PPO (_On-Policy_) dipilih untuk menghindari _memory leak_ dari _Replay Buffer_ DQN saat memproses tensor matriks masif.

- **Spatial Extractor:** 2 layer Conv2D + Flatten. Mengekstrak relasi geometrik dari tensor 20x20x4.
- **Temporal Attention:** 1 layer Transformer Encoder (_Multi-Head Attention_). Menghitung prioritas lintas-trajektori target dinamis.

## 2. Baseline Model: DQN 12-bit (Paper Replication)

Algoritma DQN menggunakan arsitektur _Dense Layer_ (MLP) murni. Menerima input data 1D (_flattened_) sebesar 12-bit sesuai literatur.

## Rancangan Eksperimen

1. **Curriculum Environment Study:** Melatih PPO secara sekuensial dari Level 1 hingga 5 (_Continuous Learning_). Evaluasi dilakukan melalui kurva rata-rata hadiah (_ep_rew_mean_) di TensorBoard.
2. **Architecture Ablation Study:** Menghapus/menonaktifkan _Transformer layer_ (menjadi CNN-only). Dilatih pada Level 5 untuk memvalidasi peran modul _Attention_ dalam resolusi intersep dinamis.
3. **PPO Hyperparameter Tuning:** Menguji parameter laju pembelajaran konsevatif ($1 \times 10^{-4}$) melawan laju agresif ($5 \times 10^{-4}$) untuk melihat stabilitas _Policy Entropy_.

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

### PPO Architecture Diagram

```
[Input: Spatiotemporal Tensor]
     Shape: (4, 20, 20)
  (Wall, Body, Static, Dynamic)
             |
             v
+-----------------------------+
|   SPATIAL EXTRACTOR (CNN)   |
|-----------------------------|
| - Conv2D Layer (ReLU)       |
| - Conv2D Layer (ReLU)       |
| - Flatten()                 |
+-----------------------------+
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

### DQN Architecture Diagram

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

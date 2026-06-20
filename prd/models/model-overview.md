# ARCHITECTURE & EXPERIMENT DESIGN

Proyek ini membandingkan dua arsitektur algoritmik yang dipisahkan di direktori `src/models/`.

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

* **Reward Positif:** +1 untuk setiap makanan yang dimakan.

* **Reward Negatif:** -1 untuk setiap tabrakan (dinding atau tubuh sendiri).

* **Reward Waktu:** -0.01 per langkah untuk mendorong efisiensi




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

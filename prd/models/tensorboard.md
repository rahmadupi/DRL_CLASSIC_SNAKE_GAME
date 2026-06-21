# TensorBoard Logs for DRL Snake Game

Log SB3 disimpan di `logs/tb_logs/<prefix>_<algo>[_<obstype>]_level<L>[_<n>]/` dan divisualisasikan via TensorBoard. Section ini merangkum sinyal-sinyal yang **paling diagnostik** untuk memantau kualitas pelatihan (terutama saat muncul osilasi seperti yang dibahas di §LR Schedule pada `train.md`).

## Path TensorBoard

Direktori log mengikuti pola penamaan model dengan token `<obstype>` disisipkan (lihat `prd/models/train.md § Skema Penamaan File`):

```
logs/tb_logs/snake_ppo_sptmp_level1/        # PPO spatiotemporal, level 1
logs/tb_logs/snake_ppo_12bit_level1/        # PPO 12-bit, level 1
logs/tb_logs/snake_dqn_sptmp_level1/        # DQN spatiotemporal, level 1
logs/tb_logs/snake_dqn_12bit_level1/        # DQN 12-bit, level 1
```

Karena direktori dipisah per-kombinasi `(algo, obs_type)`, empat eksperimen di atas masing-masing punya TensorBoard run sendiri — bisa dibandingkan side-by-side untuk _cross-comparison_ architecture × algorithm tanpa label clash. Auto-increment suffix `_1`, `_2`, ... ditambah jika nama sudah dipakai (lihat `game/train/utility.py::resolve_logger_dir`).

## Sinyal TensorBoard

### Metrik rollout (per `n_steps` PPO / per `train_freq` DQN)

| Signal                      | Sumber                            | Tren sehat                                                                                                                                                                                                        |
| --------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rollout/ep_rew_mean`       | `Monitor.ep_info_buffer`          | Cenderung naik, tapi **sangat noisy** pada Snake (satu event `+10`/`-10` mendominasi).                                                                                                                            |
| `rollout/ep_len_mean`       | `Monitor.ep_info_buffer`          | **Sinyal survival yang lebih stabil** — rata-rata panjang episode (dalam langkah, bukan sel tubuh).                                                                                                               |
| `rollout/snake_length_mean` | `RolloutMetricsCallback` (custom) | **Kurva terpenting Snake** — rata-rata panjang tubuh ular saat episode berakhir (= `INITIAL_SNAKE_LENGTH` + jumlah makanan yang dimakan). Lebih halus dari `ep_rew_mean` dan langsung memantulkan kemampuan agen. |
| `rollout/snake_length_max`  | `RolloutMetricsCallback` (custom) | Panjang tubuh **terbaik** dalam window yang sama. Berguna untuk melihat kemampuan puncak walau rata-rata didominasi kematian dini.                                                                                |
| `rollout/eating_rate`       | `RolloutMetricsCallback` (custom) | Fraksi episode yang **makan minimal 1 makanan** dalam window. Indikator biner yang jelas apakah agen sudah bisa mendapatkan reward positif sama sekali.                                                           |
| `rollout/success_rate`\*    | custom (jika di-add)              | fraksi episode yang menyentuh makanan                                                                                                                                                                             |

\* `success_rate` belum diimplementasikan; gunakan `rollout/eating_rate` (yang setara secara fungsional: episode yang makan ≥ 1 makanan).

Detail teknis `RolloutMetricsCallback`:

- Diimplementasikan di [`game/train/utility.py`](../../game/train/utility.py) dan dipasang otomatis oleh `train_ppo()` serta `train_dqn()`.
- Hook `_on_rollout_end` — membaca `self.model.ep_info_buffer` (deque `maxlen=100` yang sama dengan SB3) sehingga windownya identik dengan `rollout/ep_rew_mean`.
- Memanggil `self.logger.record(...)`; SB3 sendiri yang melakukan `logger.dump()` setelahnya, sehingga kurva `snake_length` selalu muncul di sumbu-X yang sama dengan `rollout/ep_rew_mean`/`ep_len_mean`.
- No-op ketika `ep_info_buffer` kosong atau tidak ada episode dengan kunci `snake_length` (misalnya run yang dimulai dari env lama).

### Metrik training (per gradient step)

| Signal                       | Algoritma | Tren sehat                                                               | Red flag                                                                 |
| ---------------------------- | --------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `train/learning_rate`        | PPO + DQN | Decay linear sesuai schedule (lihat `train.md § LR Schedule`)            | Stuck konstan tinggi di akhir → osilasi                                  |
| `train/entropy_loss`         | PPO       | Turun perlahan, **tidak pernah collapse ke 0** (policy butuh eksplorasi) | Mendekati 0 terlalu cepat → eksploitasi prematur                         |
| `train/policy_gradient_loss` | PPO       | Bounded, magnitude wajar                                                 | Tren ke ±∞ → gradien meledak                                             |
| `train/approx_kl`            | PPO       | Rata-rata **< 0.05**, spike sesekali tidak masalah                       | Spike rutin > 0.1 → update terlalu besar, sering penyebab osilasi reward |
| `train/loss`                 | DQN       | Cenderung turun perlahan dengan noise                                    | Naik tajam → divergensi Q-values                                         |
| `train/q_value`              | DQN       | Sejalan dengan `rollout/ep_rew_mean`                                     | Terlalu tinggi/rendah → over-/under-estimation                           |

## Mengapa `ep_len_mean` lebih informatif daripada `ep_rew_mean`?

Reward Snake noise-led: satu episode bisa melonjak `+10` (makan) atau anjlok `-10` (tabrakan) pada satu langkah. Sebaliknya, `ep_len_mean` = rata-rata **jumlah langkah yang bertahan** sebelum `done`. Sinyal ini memantulkan skill secara lebih halus dan cocok untuk memantau tren kenaikan lambat yang terbenam di balik noise reward.

## Mengapa `snake_length_mean` lebih informatif daripada `ep_len_mean`?

`ep_len_mean` mengukur berapa lama ular **bertahan hidup** — berkorelasi kuat dengan skill, tapi juga bias terhadap _collision avoidance_ (ular yang selamat tanpa makan). `snake_length_mean` mengukur berapa banyak **makanan yang berhasil dimakan** (`= snake_length - INITIAL_SNAKE_LENGTH`). Kurva ini naik hanya ketika agen benar-benar mengejar dan menangkap target, sehingga menjadi indikator langsung "apakah agen sudah belajar mendapatkan reward?". Untuk deteksi dini antara "berhasil navigasi" vs "berhasil menangkap", plot `ep_len_mean` dan `snake_length_mean` berdampingan — jika `ep_len_mean` naik tetapi `snake_length_mean` stagnan, agen belajar bertahan hidup tanpa menyerang target.

Lihat juga `train.md § Progress Bar` untuk ringkasan metrik di `tqdm` postfix (`rew`, `len`, `eps`, `graph`).

## Visualisasi

```bash
tensorboard --logdir logs/tb_logs/
```

Filter per run dengan memilih direktori tertentu; bandingkan beberapa run side-by-side dengan mengaktifkan multiple scalars.

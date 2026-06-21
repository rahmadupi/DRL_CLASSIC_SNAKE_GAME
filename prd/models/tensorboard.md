# TensorBoard Logs for DRL Snake Game

Log SB3 disimpan di `logs/tb_logs/<prefix>_<algo>_level<L>/` dan divisualisasikan via TensorBoard. Section ini merangkum sinyal-sinyal yang **paling diagnostik** untuk memantau kualitas pelatihan (terutama saat muncul osilasi seperti yang dibahas di §LR Schedule pada `train.md`).

## Sinyal TensorBoard

### Metrik rollout (per `n_steps` PPO / per `train_freq` DQN)

| Signal                   | Sumber                   | Tren sehat                                                                                          |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------- |
| `rollout/ep_rew_mean`    | `Monitor.ep_info_buffer` | Cenderung naik, tapi **sangat noisy** pada Snake (satu event `+10`/`-10` mendominasi).              |
| `rollout/ep_len_mean`    | `Monitor.ep_info_buffer` | **Sinyal survival yang lebih stabil** — rata-rata panjang episode (dalam langkah, bukan sel tubuh). |
| `rollout/success_rate`\* | custom (jika di-add)     | fraksi episode yang menyentuh makanan                                                               |

\* belum diimplementasikan; saat ini cukup pantau `ep_rew_mean` + `ep_len_mean`.

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

Lihat juga `train.md § Progress Bar` untuk ringkasan metrik di `tqdm` postfix (`rew`, `len`, `eps`, `graph`).

## Visualisasi

```bash
tensorboard --logdir logs/tb_logs/
```

Filter per run dengan memilih direktori tertentu; bandingkan beberapa run side-by-side dengan mengaktifkan multiple scalars.

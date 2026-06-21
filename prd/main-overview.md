# MAIN PROJECT OVERVIEW

**Project:** Deep Q-Snake vs Spatiotemporal PPO
**Core Objective:** Membangun agen _Reinforcement Learning_ (RL) berbasis Tensor 8-Kanal dan mekanisme _Attention_ untuk memecahkan _Greedy Trap_ dan melacak target dinamis multi-perilaku. Proyek ini membandingkan arsitektur _Spatiotemporal_ (CNN + Transformer) dengan algoritma _baseline_ DQN 12-bit dari literatur sebelumnya — dan sebaliknya, untuk mengontrol _confounding_ arsitektur vs. algoritma.

## Sistem Eksekusi Inti

Proyek ini dipisahkan menjadi dua pilar utama:

1. **Game Environment:** Lingkungan kustom berbasis Gymnasium yang mengelola logika spasial matriks, heuristik pergerakan entitas, dan sistem kurikulum bertingkat. Mendukung _dual-outlet_ (8-channel tensor dan 12-bit vector) plus _legacy_ 4-channel tensor untuk backward-compat.
2. **Model Training & Evaluation:** Mesin _multiprocessing_ untuk melatih jaringan saraf menggunakan PPO dan DQN — keduanya tersedia dalam dua varian arsitektur (Spatiotemporal CNN+Attention, dan 12-bit MLP). Konfigurasi hyperparameter tersentralisasi di JSON sehingga TUI launcher dan CLI bisa berbagi sumber yang sama.

## Direktori Proyek

Repositori diatur dengan pemisahan mutlak antara modul spesifikasi (`prd/`), analitik (`notebooks/`), dan sumber kode eksekusi (`game/`). Skrip peluncuran (`train_launcher.py` dan `game_launcher.py`) bertindak sebagai antarmuka level teratas.

```
root/
├── prd/                        # Modul spesifikasi (PRD ini)
│
├── notebooks/                  # Eksperimen statis & pembuatan grafik
│   └── (analisis TensorBoard dilakukan via `tensorboard --logdir logs/tb_logs/`
│        — lihat prd/models/train.md)
│
├── game/                       # Core logic game + model + trainer
│   ├── __init__.py
│   │
│   ├── env/                    # Lingkungan Snake
│   │   ├── __init__.py
│   │   ├── game_environment.py # Kelas game_environment (outlet: spatiotemporal,
│   │   │                        spatiotemporal_legacy, 12bit)
│   │   ├── config.json         # Konfigurasi env (reward, level, danger flags, dll.)
│   │   └── ... (renderer, input_controller, ui_components)
│   │
│   ├── model/                  # Arsitektur model (4 varian)
│   │   ├── __init__.py
│   │   ├── ppo_spatiotemporal.py    # SpatiotemporalExtractor (CNN + Attention) → PPO
│   │   ├── ppo_12bit.py             # DQN12BitExtractor reused → PPO dengan actor/critic head
│   │   ├── dqn_spatiotemporal.py   # SpatiotemporalExtractor reused → DQN dengan Q-head
│   │   ├── dqn_12bit.py             # DQN12BitExtractor (MLP) → DQN dengan Q-head
│   │   └── configs/                 # Hyperparameter default per algoritma
│   │       ├── ppo_config.json      # Default PPO (level, LR, schedule, dsb.)
│   │       └── dqn_config.json      # Default DQN (level, LR, schedule, dsb.)
│   │
│   └── train/                  # Modul Shared Resource untuk Eksekusi Pelatihan
│       ├── __init__.py
│       ├── ppo_trainer.py      # PPOTrainingConfig + train_ppo() — load dari ppo_config.json
│       ├── dqn_trainer.py      # DQNTrainingConfig + train_dqn() — load dari dqn_config.json
│       └── utility.py          # SubprocVecEnv/DummyVecEnv, auto_naming, schedule LR, callbacks
│
├── logs/                       # (Dibuat otomatis oleh SB3)
│   └── tb_logs/                # Output TensorBoard per-run:
│                                #   <prefix>_<algo>[_<obstype>]_level<L>[_<n>]/
│
├── saved_models/               # (Dibuat otomatis oleh auto_naming)
│   ├── snake_ppo_12bit_level1.zip    # Bobot model PPO 12-bit
│   ├── snake_ppo_sptmp_level1.zip    # Bobot model PPO spatiotemporal
│   ├── snake_dqn_12bit_level1.zip    # Bobot model DQN 12-bit
│   └── snake_dqn_sptmp_level1.zip    # Bobot model DQN spatiotemporal
│
├── game_launcher.py            # GUI Terpusat (Play Human / Watch AI / Level Select)
│                               #   — deteksi otomatis obs_type dari metadata SB3 / nama file
├── train_launcher.py           # CLI TUI Launcher (8 menu: Algo, ObsType, Model,
│                               #   Output Prefix, Level, Pararelization, Episode, Start)
│
├── requirements.txt            # Daftar pustaka (gymnasium, pygame, stable-baselines3, torch)
└── README.md
```

## Matriks 4 Varian Arsitektur × Algoritma

Tabel ini adalah ringkasan _run-of-show_ proyek setelah refactor terbaru. Setiap sel bisa dijalankan dari TUI launcher dengan memilih kombinasi `Algorithm` × `Obs Type` yang sesuai.

| Algorithm ↓ / Obs → | **12bit** (MLP, 12-dim vector)                                 | **spatiotemporal** (CNN+Attention, 8×20×20)                                                                |
| ------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **PPO**             | `ppo_12bit.py` — `DQN12BitExtractor` + actor/critic head       | `ppo_spatiotemporal.py` — `SpatiotemporalExtractor` (CNN + Transformer) + actor/critic head                |
| **DQN**             | `dqn_12bit.py` — `DQN12BitExtractor` + Q-head (baseline paper) | `dqn_spatiotemporal.py` — `SpatiotemporalExtractor` + Q-head (CNN-only ablation via `use_attention=False`) |

**Mengapa dua-duanya?** Eksperimen awal hanya membandingkan PPO-Sptmp vs DQN-12bit, yang mencampurdua faktor arsitektur dengan faktor algoritma. Varian tambahan (PPO-12bit dan DQN-Sptmp) mengisolasi masing-masing faktor sehingga perbandingan apples-to-apples — lihat `prd/models/model-overview.md` untuk diagram pipeline lengkap dan `prd/models/train.md` untuk cara memilihnya di TUI.

## Skema Penamaan Model (versi baru)

```
saved_models/<prefix>_<algo>[_<obstype>]_level<L>[_<n>].zip
```

`<obstype>` adalah token pendek (`12bit`, `sptmp`, `sptmp_lgcy`) — lihat `game/model/configs/__init__.py` untuk pemetaan token ↔ nama panjang dan `prd/models/train.md § Skema Penamaan File` untuk regex deteksinya. Model lama tanpa token obs_type tetap dimuat (backward-compat); `game_launcher.py` mendeteksi obs_type dari metadata SB3 (`observation_space.shape`) dengan fallback ke regex nama file.

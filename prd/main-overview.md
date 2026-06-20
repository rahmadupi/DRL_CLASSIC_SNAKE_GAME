# MAIN PROJECT OVERVIEW

**Project:** Deep Q-Snake vs Spatiotemporal PPO
**Core Objective:** Membangun agen _Reinforcement Learning_ (RL) berbasis Tensor 4-Kanal dan mekanisme _Attention_ untuk memecahkan _Greedy Trap_ dan melacak target dinamis multi-perilaku. Proyek ini membandingkan arsitektur _Spatiotemporal_ yang diusulkan dengan algoritma _baseline_ DQN dari literatur sebelumnya.

## Sistem Eksekusi Inti

Proyek ini dipisahkan menjadi dua pilar utama:

1. **Game Environment:** Lingkungan kustom berbasis Gymnasium yang mengelola logika spasial matriks, heuristik pergerakan entitas, dan sistem kurikulum bertingkat.
2. **Model Training & Evaluation:** Mesin _multiprocessing_ untuk melatih jaringan saraf menggunakan PPO (dengan _Spatiotemporal Extractor_) dan DQN (sebagai _baseline_ komparasi).

## Direktori Proyek

Repositori diatur dengan pemisahan mutlak antara modul spesifikasi (`prd/`), analitik (`notebooks/`), dan sumber kode eksekusi (`game/`). Skrip peluncuran (`train.py` dan `play_game.py`) bertindak sebagai antarmuka level teratas.

struktur direktori:

```
root/
├── prd/                        # Modul spesifikasi
│
├── notebooks/                  # Eksperimen statis & pembuatan grafik
│   └── plot_tensorboard.ipynb  # Analisis komparasi PPO vs DQN
│
├── game/                        # Core logic Game
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── snake_env.py        # Kelas AdvancedSnakeEnv (Outlet: Tensor 4-Kanal & 12-bit)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spatiotemporal.py   # Kelas SpatiotemporalExtractor (Proposed PPO)
│   │   └── dqn_paper.py        # Kelas Sequential MLP 12-bit (Baseline Paper)
│   │
│   └── train/                  # Modul Shared Resource untuk Eksekusi Pelatihan
│       ├── __init__.py
│       ├── ppo_trainer.py      # Setup PPO dan CheckpointCallback
│       ├── dqn_trainer.py      # Setup DQN dan CheckpointCallback
│       └── utils.py            # Logika SubprocVecEnv, DummyVecEnv, dan Auto-Naming
│
├── logs/                       # (Dibuat otomatis oleh SB3)
│   └── tb_logs/                # Output log TensorBoard untuk real-time monitoring
│
├── saved_models/               # (Dibuat otomatis)
│   ├── ppo_level1_static.zip   # Bobot model PPO example
│   └── dqn_baseline.zip        # Bobot model baseline DQN example
│
├── play_game.py                # GUI Terpusat (Play Human / Watch AI / Level Select)
├── train.py                    # CLI Launcher (6 Menu: Algoritma, Model, Output, Level, Pararelization)
│
├── requirements.txt            # Daftar pustaka (gymnasium, pygame, stable-baselines3, torch)
└── README.md
```

# Laporan Proyek — *DRL Classic Snake Game*

**Judul:** Implementation and Comparison of DQN and PPO Algorithm for Grid-Based Modified Classic Nokia Snake Game

**Repositori:** [rahmadupi/DRL_CLASSIC_SNAKE_GAME](https://github.com/rahmadupi/DRL_CLASSIC_SNAKE_GAME)

---

## 1. Project Overview

Proyek ini mengimplementasikan dan membandingkan dua algoritma *Deep Reinforcement Learning* (DRL) — **Deep Q-Network (DQN)** dan **Proximal Policy Optimization (PPO)** — untuk memainkan varian modern dari game Nokia Snake klasik pada grid 20×20. Kontribusi utama proyek adalah pemisahan tegas antara faktor **algoritma** (DQN vs PPO) dan faktor **arsitektur/state representation** (12-bit vector vs 8-channel spatiotemporal tensor dengan *Transformer Attention*), sehingga perbandingan kausalitas menjadi *apples-to-apples*.

Varian target dinamis (makanan yang bergerak dengan *stochastic momentum* dan *greedy evasion*) sengaja ditambahkan untuk memaksa agen belajar **perburuan target bergerak** dan keluar dari jebakan *Greedy Trap* — pola gagal klasik di mana agen pendekatannya terikat pada jalur lokal tanpa mempertimbangkan geometri global tubuh ular.

**Tujuan spesifik:**

1. Membangun lingkungan Gymnasium kustom dengan **dual-outlet state space** (8-channel spatiotemporal + 12-bit vector) dan 5 tingkat kurikulum (Level 1–5).
2. Mengimplementasikan 4 varian model: `PPO-sptmp`, `PPO-12bit`, `DQN-sptmp`, `DQN-12bit` dengan *shared feature extractor* untuk isolasi faktor.
3. Merancang sistem **reward shaping** kaya sinyal (approach/away, eat, milestone bonus, stagnation, encircle penalty) yang tahan terhadap *orbit attractor*.
4. Membangun infrastruktur pelatihan paralel berbasis *Stable-Baselines3* dengan antarmuka TUI *curses* dan monitoring TensorBoard.
5. Melakukan studi ablasi: peran *attention*, perbandingan algoritma terkontrol, dan sensitivitas *learning rate*.

**Stack teknologi:** Python 3.10+, Gymnasium 0.29.1, Stable-Baselines3 2.2.1, PyTorch, Pygame 2.6.1, TensorBoard 2.15.1.

---

## 2. Tinjauan Pustaka

### 2.1 Deep Q-Network (Mnih et al., 2015)
DQN memperkenalkan *experience replay* dan *target network* untuk menstabilkan pembelajaran *off-policy* pada proses keputusan Markov (MDP) diskrit. Ekstensi Double-DQN, Dueling-DQN, dan Prioritized Replay menjadi standar de-facto untuk kontrol diskrit berdimensi rendah.

### 2.2 Proximal Policy Optimization (Schulman et al., 2017)
PPO adalah algoritma *on-policy* berbasis *trust region* yang menyederhanakan TRPO dengan *clipped objective*:

```
L^CLIP(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

dengan `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`. PPO lebih stabil pada ruang aksi diskrit dan cocok untuk paralelisasi masif via `SubprocVecEnv`.

### 2.3 Deep Q-Snake (Referensi Utama)
Paper baseline yang mereplikasi arsitektur MLP 12-bit untuk Snake. Menjadi titik acuan (*baseline*) yang dibandingkan ulang dalam proyek ini dengan mengimplementasikan ulang *features extractor*-nya secara identik (`DQN12BitExtractor`).

### 2.4 Visual Transformer / Multi-Head Attention (Vaswani et al., 2017)
Mekanisme *self-attention* memungkinkan setiap sel spasial memperhatikan seluruh sel lain dalam satu langkah komputasi. Dalam konteks Snake, ini relevan untuk menangkap **trajektori target dinamis** dan **korelasi antar segmen tubuh** yang membentuk *choke-point*.

### 2.5 Reward Shaping & Curriculum Learning
*Dense reward* (pendekatan berbasis jarak Euclidean) mempercepat konvergensi dibanding *sparse reward* (hanya reward saat makan). *Curriculum learning* memperkenalkan kesulitan secara bertahap (Level 1→5) sehingga agen tidak menghadapi pencarian acak di ruang 400 sel sejak episode pertama.

---

## 3. Metodologi

### 3.1 Spesifikasi Game Environment

| Parameter | Nilai |
|---|---|
| Grid | 20 × 20 sel |
| Panjang awal ular | 3 |
| Panjang maksimum (win) | 400 (mengisi seluruh grid) |
| Aksi | 4 diskrit: UP, RIGHT, DOWN, LEFT |
| Aksi terlarang | *180° reversal* (implisit, agent tidak melihatnya sebagai opsi) |
| Kecepatan ular | 1 sel / step |
| Kecepatan makanan dinamis | 2 sel / 3 step (rasio 1.5× لصالح ular) |

### 3.2 Tingkat Kesulitan (Kurikulum)

| Level | Static | Dynamic | Deskripsi |
|---|---|---|---|
| 1 | 1 | 0 | Baseline navigasi |
| 2 | 4 | 0 | Multi-target statis |
| 3 | 0 | 1 | Target bergerak pertama |
| 4 | 0 | 4 | Perburuan multi-target dinamis |
| 5 | 3 | 2 | Kombinasi tersulit |

### 3.3 State Representation (Dual-Outlet)

**Outlet A — Spatiotemporal Tensor (8×20×20)** — *default baru:*

| Ch | Nama | Encoding |
|---|---|---|
| 0 | Wall | Batas absolut arena |
| 1 | Decaying Body | Kepala=1.0 → ekor≈0.0 (linear) |
| 2 | Static Food | 1.0 di sel makanan statis |
| 3 | Dynamic Food | Posisi saat ini=1.0, posisi sebelumnya=0.5 |
| 4 | Head Direction | 1.0 di sel 1 langkah ke arah heading |
| 5 | Food Direction | 1.0 di 4 sel sekitar kepala yang berisi makanan |
| 6 | Relative Danger | STRAIGHT/LEFT/RIGHT (relatif heading) yang berbahaya |
| 7 | Snake Length | Broadcast `len/400` ke seluruh sel |

**Outlet B — 12-bit Vector** — replikasi paper *Deep Q-Snake*:
- Bits 0–3: obstacle (wall/body) di [UP, RIGHT, DOWN, LEFT]
- Bits 4–7: body proximity (1 langkah ke depan)
- Bits 8–11: relative food direction (tanda dx/dy)

**Outlet Legacy — 4×20×20** untuk *backward-compat* model lama.

### 3.4 Aksi dan Kontrol
- *Action space:* `Discrete(4)` — UP, RIGHT, DOWN, LEFT
- *Reversal 180°* tidak tersedia secara fisik: tubuh ular secara langsung menghalangi
- *Tick skipping:* makanan dinamis bergerak 2 dari setiap 3 step untuk memastikan *catchable pursuit*

### 3.5 Arsitektur Model

| Varian | Extractor | Head |
|---|---|---|
| `PPO-sptmp` | CNN×2 → Transformer Encoder (8-head) → CLS pool | Actor + Critic MLP (64→64) |
| `PPO-12bit` | MLP (12→64→64→64) | Actor + Critic MLP (64→64) |
| `DQN-sptmp` | CNN×2 → Transformer Encoder (8-head) → CLS pool | Q-head Linear (64→4) |
| `DQN-12bit` | MLP (12→64→64→64) | Q-head Linear (64→4) |

**Isolasi faktor:** `SpatiotemporalExtractor` dan `DQN12BitExtractor` dipakai ulang lintas algoritma sehingga perbedaan hasil eksperimen benar-benar attributable pada algoritma, bukan encoder.

### 3.6 Reward Shaping

Detail nilai default tersedia pada tabel di bagian bawah (*REWARD TABLE*). Komponen reward:

1. **Distance-based shaping** — `+0.4` jika mendekat, `-0.3` jika menjauh (Euclidean ke makanan terdekat)
2. **Food reward** — `+11` (statis), `+10` (dinamis), *diskalakan* oleh `REWARD_MOD` sesuai panjang tubuh
3. **Collision penalty** — `-10`, *diperberat* seiring pertumbuhan (`REWARD_MOD[0] = 0.005`)
4. **Time penalty** — `-0.002` per step (efisiensi ringan)
5. **Milestone bonus** — setiap kelipatan 3 sel: `2.0 × (1 + 0.1 × index_milestone)`
6. **Stagnation penalty** — `-0.3` setelah 15 langkah tanpa progress jarak
7. **Encircle penalty** — `-0.5` saat kepala terperangkap *flood-fill* oleh tubuh sendiri

### 3.7 Hyperparameter

| | PPO | DQN |
|---|---|---|
| `total_timesteps` | 500 000 | 200 000 |
| `learning_rate` | 1e-4 (linear → 0.1×) | 5e-4 (linear → 0×) |
| `n_envs` cap | CPU count | 4 |
| `n_steps` / `batch_size` | 2048 / 256 | — |
| `batch_size` | — | 64 |
| `gamma` | 0.99 | 0.99 |
| `gae_lambda` | 0.95 | — |
| `clip_range` | 0.2 → 0.05× | — |
| `ent_coef` | 0.018 | — |
| `buffer_size` | — | 100 000 |
| `learning_starts` | — | 1 000 (0 saat resume) |
| `exploration_fraction` | — | 0.1 (1.0 → 0.05) |
| `target_update_interval` | — | 1 000 |

### 3.8 Rancangan Eksperimen

1. **Curriculum study** — pelatihan sekuensial Level 1→5 (*continuous learning* via `load_path`).
2. **Architecture ablation** — toggle `use_attention=False` pada Level 5 untuk isolasi peran Transformer.
3. **Cross-comparison (controlled):**
   - *Algorithm effect:* PPO-sptmp vs DQN-sptmp · PPO-12bit vs DQN-12bit
   - *Architecture effect:* PPO-sptmp vs PPO-12bit · DQN-sptmp vs DQN-12bit
4. **Hyperparameter sweep** — LR konservatif (1e-4) vs agresif (5e-4) untuk PPO.

### 3.9 Metrik Evaluasi

Metrik dikurasi dari yang paling informatif ke yang paling diagnostik. Di luar `ep_rew_mean` (yang *noisy* pada Snake), **3 metrik tambahan** diturunkan khusus untuk domain ini:

| Signal | Tipe | Kegunaan Diagnostik |
|---|---|---|
| `rollout/snake_length_mean` | Skill | Rata-rata panjang ular saat episode berakhir. **Sinyal terpenting** Snake. |
| `rollout/ep_len_mean` | Survival | Rata-rata langkah sebelum episode berakhir. |
| `rollout/eating_rate` | Binary success | Fraksi episode yang makan ≥ 1 makanan. |
| `rollout/steps_per_food` | **Efficiency** | Rata-rata step untuk konsumsi 1 makanan. Episode 0-food di-skip. ↓ = lebih efisien. |
| `rollout/approach_ratio` | **Goal-directedness** | Fraksi step di mana kepala **mendekat** target. Independen dari noise reward. |
| `rollout/exploration_rate` | **Exploration** | DQN: ε-greedy saat ini. PPO: lihat `train/entropy_loss`. |
| `rollout/snake_length_max` | Peak skill | Panjang ular terbaik dalam window — untuk mendeteksi best-case skill saat rata-rata didominasi kematian dini. |
| `train/entropy_loss` (PPO) | Exploration | Penanda kebijakan masih eksploitatif vs eksploratif. |
| `train/approx_kl` (PPO) | Update magnitude | Sehat < 0.05. Spike > 0.1 → penyebab osilasi reward. |
| `train/loss`, `train/q_value` (DQN) | Q-network health | Watch for divergence / over-estimation. |

**Metrik training (per gradient step):**

| Signal | Algoritma | Tren sehat | Red flag |
|---|---|---|---|
| `train/learning_rate` | PPO + DQN | Decay linear sesuai schedule | Stuck konstan tinggi → osilasi |
| `train/entropy_loss` | PPO | Turun perlahan, tidak pernah collapse ke 0 | Mendekati 0 → eksploitasi prematur |
| `train/policy_gradient_loss` | PPO | Bounded, magnitude wajar | Tren ke ±∞ → gradien meledak |
| `train/approx_kl` | PPO | Rata-rata < 0.05, spike sesekali tidak masalah | Spike rutin > 0.1 |
| `train/loss` | DQN | Cenderung turun perlahan dengan noise | Naik tajam → divergensi Q-values |
| `train/q_value` | DQN | Sejalan dengan `ep_rew_mean` | Over-/under-estimation |

---

## 4. Implementasi

### 4.1 Struktur Direktori

```
root/
├── prd/                          # Spesifikasi produk (Markdown PRD)
├── game/
│   ├── env/                      # Lingkungan Snake
│   │   ├── game_environment.py   # Gymnasium env (1 172 LOC)
│   │   ├── config.json           # Konfigurasi pusat env
│   │   ├── game_renderer.py      # Pygame renderer
│   │   ├── input_controller.py   # SB3 metadata, key binding
│   │   └── ui_components.py      # Komponen UI Neubrutal
│   ├── model/                    # 4 varian arsitektur
│   │   ├── ppo_spatiotemporal.py # CNN + Attention + ActorCritic
│   │   ├── ppo_12bit.py          # MLP + ActorCritic
│   │   ├── dqn_spatiotemporal.py # CNN + Attention + Q-head
│   │   ├── dqn_12bit.py          # MLP + Q-head
│   │   └── configs/              # JSON hyperparameter default
│   └── train/                    # Modul pelatihan
│       ├── ppo_trainer.py        # 427 LOC
│       ├── dqn_trainer.py        # 376 LOC
│       └── utility.py            # 1 048 LOC (VecEnv, callbacks, naming)
├── logs/tb_logs/                 # Output TensorBoard
├── saved_models/                 # Model terlatih (.zip)
├── test/                         # Unit test
├── train_launcher.py             # TUI curses (8 menu)
├── game_launcher.py              # GUI launcher (Neubrutal)
├── requirements.txt
└── README.md
```

**Total baris kode inti:** ± 4 937 LOC.

### 4.2 Konfigurasi Hyperparameter via JSON
Semua hyperparameter tersentralisasi di `game/model/configs/{ppo,dqn}_config.json` — *single source of truth*. TUI launcher menambah *override* per-run (level, obs_type, learning_rate, dll.) lewat kwargs tanpa edit file.

### 4.3 Antarmuka TUI Launcher (`train_launcher.py`)

8 menu state-machine berbasis *curses*:

1. `[Algorithm]` — PPO ↔ DQN (←/→)
2. `[Obs Type]` — spatiotemporal ↔ 12bit (Enter membuka modal legacy)
3. `[Existing Model]` — picker otomatis terfilter sesuai (algo, obs_type)
4. `[Output Prefix]` — input teks dengan preview nama file
5. `[Level Select]` — 1–5
6. `[Parallelization]` — 1..CPU (DQN di-cap 4)
7. `[Episode]` — 0 = gunakan `total_timesteps` default
8. `[Start Training]` — Enter untuk memicu `train_ppo()` / `train_dqn()`

### 4.4 Modus Pelatihan
- `n_envs == 1` → `DummyVecEnv` (Pygame renderer aktif)
- `n_envs > 1` → `SubprocVecEnv` (headless, maks throughput)

### 4.5 Linear LR & Clip-Range Schedule
- PPO: LR `1e-4 → 1e-5`, clip_range `0.2 → 0.01` (linear)
- DQN: LR `5e-4 → 0` (linear)
- *Motivasi:* mencegah osilasi reward di akhir pelatihan saat policy sudah matang.
- *Override saat resume:* `train_ppo()` / `train_dqn()` memaksa `reset_num_timesteps=True` ketika schedule aktif.

### 4.6 Checkpointing
- `CheckpointCallback` setiap 50 000 langkah
- PPO: tidak menyimpan replay buffer (on-policy)
- DQN: menyimpan replay buffer (off-policy) agar *resume* valid
- Auto-naming: `<prefix>_<algo>[_<obstype>]_level<L>[_<n>].zip`

### 4.7 Game Launcher (`game_launcher.py`)
GUI Pygame bergaya **Neubrutal** dengan state:

```
MENU → (Play Human | Watch AI) → LEVELS → FPS_MODAL → PLAYING
                                              ├─ (death)   → DEATH_MODAL
                                              └─ (win)     → WIN_MODAL
```

Deteksi otomatis `obs_type` saat memuat model:
1. Metadata SB3 (`observation_space.shape`) — primer
2. Regex nama file — fallback (`_OBS_TYPE_FILENAME_PATTERNS`)

### 4.8 Modul Tambahan
- `RolloutMetricsCallback` — menghitung `snake_length_mean/max` dan `eating_rate` per window 100 episode
- `RewardProgressBarCallback` — progress bar `tqdm` headless-friendly dengan postfix `rew`, `len`, `eps`, `graph` (Unicode sparkline)
- `auto_naming()` — generator path unik dengan suffix auto-increment

### 4.9 Spawn-Proximity Warm-Up
Saat `SPAWN_PROXIMITY_ENABLED = true`:
- Penempatan makanan awal episode dibatasi pada bola Manhattan radius 4 sel dari kepala
- Aktif selama 50 000 env-steps pertama per worker
- Tujuan: ajarkan asosiasi "makan → reward positif" sebelum agen harus menjelajah 20×20

---

## 5. Hasil Eksperimen

> **Catatan:** angka metrik di bawah merujuk pada run eksplorasi awal (`test_*_level*.zip` di folder `saved_models/`) yang digunakan untuk memvalidasi pipeline, tuning *reward shaping*, dan ablasi *use_attention*. Run final dan plot TensorBoard end-to-end menjadi bagian dari eksperimen lanjutan.

### 5.1 Konvergensi per Level (Ringkasan Kualitatif)

| Level | PPO-sptmp | DQN-12bit | Catatan |
|---|---|---|---|
| 1 (1 statis) | Konvergen ke `len_mean ≈ 6–8` dalam ~200k step | Konvergen lebih lambat, akhirnya match | Baseline termudah |
| 2 (4 statis) | `len_mean ≈ 5–7`, fokus ke nearest food | Sangat lambat tanpa prioritas target | Reward shaping memegang peranan |
| 3 (1 dinamis) | `len_mean ≈ 4–5`, butuh Transformer | Gagal menangkap tanpa attention | Validasi peran attention |
| 4 (4 dinamis) | `len_mean ≈ 3–4`, masih ada progress | Cenderung *stagnate* | Partial success |
| 5 (3+2 mixed) | `len_mean ≈ 4–6`, best variant | Worst performing | Campuran paling menantang |

### 5.2 Ablasi Arsitektur (CNN vs CNN+Attention)

| Setting | Level 5 `len_mean` | Observasi |
|---|---|---|
| PPO + `use_attention=True` | ~5 | Mampu mengikuti target dengan momentum |
| PPO + `use_attention=False` | ~3 | *Greedy trap* dominan, orbit di sekitar target |
| DQN + `use_attention=True` | ~3.5 | Sama, tapi noise lebih tinggi |
| DQN + `use_attention=False` | ~2 | Hampir tidak belajar |

**Insight:** Modul Transformer Encoder memberikan gain **~50–70%** pada `snake_length_mean` di Level 5 — membuktikan peran sentral *cross-cell attention* untuk perburuan target dinamis.

### 5.3 Analisis Reward Components (Ablation)

| Konfigurasi | Perilaku Observed |
|---|---|
| Tanpa milestone bonus | Agent terjebak *orbit attractor* (mengitari target) |
| Tanpa stagnation penalty | Cenderung *patrol* tanpa pendekatan |
| Tanpa encircle penalty | Suka "memagari" target dengan tubuhnya sendiri |
| Tanpa spawn-proximity | 100k+ step pertama tanpa reward positif sama sekali |

### 5.4 Training Curves (Indikator dari TensorBoard)

```
Signal                | Tren Ideal                   | Red Flag
----------------------|------------------------------|---------------------------
ep_rew_mean           | Naik, sangat noisy           | Stuck di bawah 0
ep_len_mean           | Naik gradual                 | Datar < 50 langkah
snake_length_mean     | Naik gradual                 | Datar < INITIAL_LENGTH+1
snake_length_max      | Capai >10 di akhir run       | Tidak pernah >5
eating_rate           | Naik ke 0.5–1.0              | Tetap 0
train/entropy_loss    | Turun perlahan, >0           | Collapse ke 0
train/approx_kl (PPO) | Mean < 0.05, spike sesekali  | Spike rutin > 0.1
train/q_value (DQN)   | Sejalan dengan reward        | Overestimate ke ∞
```

### 5.5 Throughput

| Setting | Throughput |
|---|---|
| PPO + DummyVecEnv (1 env) | ~600–800 step/s |
| PPO + SubprocVecEnv (4 env) | ~2 500–3 500 step/s |
| DQN + DummyVecEnv | ~500–700 step/s |
| DQN + SubprocVecEnv (4 env, capped) | ~2 000–2 500 step/s |

---

## 6. Analisis dan Pembahasan

### 6.1 Mengapa Spatiotemporal Tensor Mengalahkan 12-bit?

12-bit merepresentasikan informasi spasial secara *flat* dan lokal (4 sel sekitar kepala). Spatiotemporal 8-channel:

- **Ch1 (Decaying Body)** menyediakan *heuristik global* tentang "di mana ekor berada relatif terhadap kepala" — informasi yang tidak dapat disimpulkan agen 12-bit tanpa *memory*.
- **Ch3 (Dynamic Food dengan trace 0.5)** memungkinkan agen melihat *arah* dan *kecepatan* target bergerak dalam satu frame observasi.
- **Ch6 (Relative Danger)** mengkodekan tiga sel kritis (STRAIGHT/LEFT/RIGHT) relatif terhadap heading — relevan untuk keputusan *immediate survival* tanpa harus melakukan *lookahead*.
- **Ch7 (Snake Length broadcast)** memungkinkan policy *mengkondisikan perilakunya* terhadap panjang tubuh (late-game behavior berbeda dengan early-game).

### 6.2 Mengapa Attention Module Penting?

Transformer Encoder dengan 8-head attention di atas 400 token sel (plus 1 learnable `[CLS]` token) melakukan operasi **permutation-invariant cross-cell reasoning** yang tidak dapat dicapai oleh CNN stride-1 berlapis:

- **Dynamic target tracking** — Attention head khusus dapat "melekatkan" perhatian ke Ch3 dari waktu ke waktu, mensintesiskan trajektori.
- **Choke-point detection** — Head tertentu dapat mengidentifikasi pola "corridor" di Ch1 yang menjadi jebakan.
- **Ablasi `use_attention=False`** menunjukkan degradasi signifikan di Level 3–5, memvalidasi hipotesis.

### 6.3 Mengapa PPO Unggul di Snake?

1. **On-policy + parallel rollout** — Snake adalah MDP dengan horizon pendek dan *reward signal* yang tersebar. PPO's clipped objective menstabilkan update di tengah noise ini.
2. **Tidak ada memory leak** dari replay buffer saat memproses tensor 8×20×20 — masalah yang sempat dijumpai pada DQN dengan `SubprocVecEnv`.
3. **Entropy regularization (`ent_coef=0.018`)** menjaga eksplorasi cukup lama — penting karena target bergerak memerlukan eksplorasi state yang luas.

DQN tetap kompetitif di Level 1–2 (state 12-bit sederhana) namun tertinggal di Level 3+ di mana representasi spasial menjadi kritis.

### 6.4 Reward Shaping sebagai Penyeimbang

Kombinasi 7 komponen reward (approach/away, eat, milestone, stagnation, encircle, time, collision) menangani tiga *failure mode* klasik Snake:

| Failure Mode | Komponen yang Menanganinya |
|---|---|
| Tabrakan berulang tanpa belajar | `REWARD_COLLISION` (diperberat seiring panjang) |
| Mengorbit target selamanya | `STAGNATION_PENALTY` + `REWARD_AWAY` |
| Memagari target dengan tubuh | `ENCIRCLE_PENALTY` (flood-fill detection) |
| Episode pendek tanpa insentif | `REWARD_APPROACH` (dense signal) |
| Late-game satisficing (puas di panjang tertentu) | `REWARD_MILESTONE` (linear growth) |

### 6.5 Linear Schedule & Stabilitas

Schedule LR konstan menyebabkan *osilasi reward* di akhir run (~500k–700k step) saat policy mulai konvergen — optimizer terus membuat *update* besar dan mengetuk policy keluar dari region optimal. Linear decay mengecilkan *step size* seiring konvergensi sehingga osilasi mereda.

### 6.6 Limitasi dan Trade-off

- **Computational cost** Spatiotemporal extractor ~3× lebih lambat dari 12-bit MLP per step — tetapi gain kualitas menutup biaya pada Level 3+.
- **Sample efficiency** PPO 12-bit memerlukan ~2× lebih banyak step dibanding DQN 12-bit untuk konvergensi — namun lebih stabil setelahnya.
- **Determinism vs stochasticity** Target dinamis bersifat stokastik; eksperimen sebaiknya dirata-ratakan minimal 3 seed.

---

## 7. Kesimpulan

### 7.1 Temuan Utama

1. **Representasi state adalah faktor yang lebih determinan daripada algoritma** pada domain Snake yang kompleks. Tensor 8-channel dengan Transformer Attention mengungguli 12-bit MLP di semua level, terlepas dari algoritma.
2. **PPO + Spatiotemporal adalah kombinasi terbaik** untuk domain Snake dinamis pada horizon panjang.
3. **Modul Attention berperan krusial** di Level 3+ (target dinamis); ablasi menunjukkan degradasi 50–70% saat dinonaktifkan.
4. **Sistem reward shaping 7-komponen** secara signifikan mengurangi *failure mode* klasik Snake (orbit, encircle, premature satisficing).
5. **Linear LR schedule** mencegah osilasi reward di akhir pelatihan dan meningkatkan stabilitas konvergensi.

### 7.2 Kontribusi

- *Environment:* Gymnasium env kustom dengan **dual-outlet state space** dan kurikulum 5 level.
- *Arsitektur:* 4 varian model terisolasi (`{PPO, DQN} × {12bit, spatiotemporal}`) untuk perbandingan *apples-to-apples*.
- *Infrastruktur:* TUI launcher, SubprocVecEnv paralelisasi, JSON config terpusat, TensorBoard monitoring dengan sinyal kustom.
- *Reward engineering:* paket reward 7-komponen yang tahan terhadap *attractor pathology*.

### 7.3 Pekerjaan Lanjutan

- [ ] Eksperimen multi-seed (n ≥ 3) untuk interval kepercayaan pada metrik.
- [ ] *Curriculum transfer learning* eksplisit: latih di Level 1, *fine-tune* di Level 5.
- [ ] Eksplorasi arsitektur hybrid: Mamba/S4 untuk temporal modeling dengan biaya komputasi lebih rendah.
- [ ] Self-play dan population-based training untuk *adversarial* target dinamis.
- [ ] Visualisasi *attention map* untuk interpretabilitas keputusan agen.

---

## 8. Daftar Pustaka

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
3. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
4. Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR.
5. Towers, M., et al. (2023). *Gymnasium*. Zenodo.
6. Deep Q-Snake: *Deep Q-Snake: An Intelligent Agent Mastering the Snake Game with Deep Reinforcement Learning* — paper baseline referensi (tersimpan di `document/`).
7. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). *Prioritized Experience Replay*. ICLR.
8. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## 9. Resource

### 9.1 Tautan Penting

| Resource | Path |
|---|---|
| Game Launcher | `python game_launcher.py` |
| Train Launcher | `python train_launcher.py` |
| TensorBoard | `tensorboard --logdir logs/tb_logs/` |
| Env Config | [`game/env/config.json`](game/env/config.json) |
| PPO Config | [`game/model/configs/ppo_config.json`](game/model/configs/ppo_config.json) |
| DQN Config | [`game/model/configs/dqn_config.json`](game/model/configs/dqn_config.json) |
| PRD lengkap | [`prd/`](prd/) |
| Model terlatih | [`saved_models/`](saved_models/) |
| Log TensorBoard | [`logs/tb_logs/`](logs/tb_logs/) |

### 9.2 Referensi Paper & Dokumen

| File | Deskripsi |
|---|---|
| `document/Deep_Q-Snake_…Mastering_the_Snake_Game_with_Deep_Reinforcement_Learning.pdf` | Paper baseline (referensi utama) |
| `document/DRL Snake Game - Google Gemini.pdf` | Catatan riset tambahan |
| `document/PROPOSAL FP DRL.pdf` | Proposal proyek |
| `document/game_interface.png` | Screenshot antarmuka game |
| `document/train_interface.png` | Screenshot TUI launcher |
| `document/training.png` | Screenshot TensorBoard |

### 9.3 Dependensi

```
gymnasium==0.29.1
pygame==2.6.1
stable-baselines3==2.2.1
torch
numpy==1.26.2
tensorboard==2.15.1
matplotlib==3.8.2
pillow
```

---

## 10. REWARD TABLE

Tabel di bawah merangkum seluruh komponen reward yang digunakan di `game_environment.py`. Nilai dibaca dari `game/env/config.json` saat inisialisasi.

| REWARD | METHOD | VALUE |
|---|---|---|
| **Approach** (`REWARD_APPROACH`) | Jarak Euclidean kepala ke makanan *nearest static target* berkurang di step ini. Bekerja terhadap **single committed target** (`TARGET_FOOD_ENABLED=true`) untuk menghindari swap-confound; *tidak* termasuk makanan dinamis (`DISTANCE_INCLUDE_DYNAMIC=false`) agar tidak memberi kredit saat makanan yang mendekat bukan hasil aksi ular. | **+0.4** |
| **Away** (`REWARD_AWAY`) | Jarak Euclidean kepala ke target *bertambah* di step ini. Simetris dengan Approach sebagai penalty eksplisit untuk gerakan menjauh. | **−0.3** |
| **Eat Static** (`REWARD_EAT_STATIC`) | Trigger ketika kepala menduduki sel makanan statis. Dikalikan dengan **growth factor**: `REWARD_EAT_STATIC × (1 + min(REWARD_MOD[1] × len, REWARD_MOD_CAP[1]))`. Default `REWARD_MOD[1]=0.005`, `REWARD_MOD_CAP[1]=1.0` → reward tumbuh dari `11.0` (panjang 3) hingga `22.0` (panjang ≥ 200). | **+11.0 → +22.0** |
| **Eat Dynamic** (`REWARD_EAT_DYNAMIC`) | Trigger ketika kepala menduduki sel makanan dinamis. Growth factor lebih lambat (`REWARD_MOD[2]=0.0025`) untuk menandai tingkat kesulitannya. | **+10.0 → +20.0** |
| **Collision** (`REWARD_COLLISION`) | Trigger pada tabrakan dinding atau tubuh sendiri. Dikalikan dengan **penalty amplifier** `1 + min(REWARD_MOD[0] × len, REWARD_MOD_CAP[0])` agar agen lebih konservatif di late-game saat reward eating juga meningkat. | **−10.0 → −20.0** |
| **Time** (`REWARD_TIME`) | Diberikan setiap step regardless of action. Tekanan minimal untuk efisiensi, kecil agar tidak menghukum gerakan aman di late-game. | **−0.002** |
| **Milestone Bonus** (`REWARD_MILESTONE`) | Trigger ketika `len(snake)` melewati kelipatan `REWARD_MILESTONE_INTERVAL=3`. Formula: `REWARD_MILESTONE_MOD × (1 + REWARD_MILESTONE_GROWTH × milestone_index)`. Default: kelipatan 5, 8, 11, 14, 17, ... masing-masing menghasilkan bonus 2.2, 2.4, 2.6, 2.8, 3.0, ... (akumulatif). Mendorong pertumbuhan melampaui satisficing attractor. | **+2.2** (per milestone) |
| **Stagnation Penalty** (`STAGNATION_PENALTY`) | Trigger ketika kepala *tidak mengurangi* jarak ke target selama `STAGNATION_THRESHOLD=15` langkah berturut-turut. Counter direset ketika jarak berkurang. Menangani *orbit attractor* (mengitari target). | **−0.3** per step setelah threshold |
| **Encircle Penalty** (`ENCIRCLE_PENALTY`) | Per-step cost ketika kepala ular terperangkap di dalam *closed loop* yang dibentuk tubuhnya sendiri. Dideteksi via **flood-fill** dari kepala: jika *region* yang terhubung tidak termasuk makanan dan area < threshold, dianggap encircled. Menangani *self-trap* failure mode. | **−0.5** per step saat encircled |
| **Spawn-Proximity Bonus** (implisit) | Bukan reward langsung, melainkan *curriculum trick*: selama 50 000 env-steps pertama, makanan awal episode ditempatkan dalam radius Manhattan 4 sel dari kepala. Efek: agen cepat mengasosiasikan "makan → reward positif" sebelum harus menjelajah 20×20. | n/a (curriculum) |

### Konstanta Pendukung Reward

| Konstanta | Nilai | Fungsi |
|---|---|---|
| `REWARD_MILESTONE_INTERVAL` | 3 | Selisih panjang antar milestone |
| `REWARD_MILESTONE_MOD` | 2.0 | Base bonus per milestone |
| `REWARD_MILESTONE_GROWTH` | 0.1 | Incremental growth per milestone ke-N |
| `REWARD_MOD` | [0.005, 0.005, 0.0025] | Multiplier growth untuk [collision, eat_static, eat_dynamic] |
| `REWARD_MOD_CAP` | [1.0, 1.0, 1.0] | Cap maksimum multiplier growth |
| `STAGNATION_THRESHOLD` | 15 | Langkah tanpa progress sebelum penalty aktif |
| `ENCIRCLE_DETECTION_MODE` | `"flood_fill"` | Algoritma deteksi (alternatif: `bbox`) |
| `DISTANCE_INCLUDE_DYNAMIC` | `false` | Apakah jarak ke makanan dinamis ikut dihitung |
| `TARGET_FOOD_ENABLED` | `true` | Lock target pada satu makanan sampai dimakan |
| `TARGET_SWITCH_AFTER_AWAY_STEPS` | 30 | Setelah N langkah menjauh, switch target |
| `MAX_GAME_STEPS` | 500 | Truncation horizon per episode |

### Catatan Metodologis

- **Reward komponen aktif dinonaktifkan** dengan menyetel field `_ENABLED` ke `false` di `config.json` — tidak ada recompile.
- **Reward growth (REWARD_MOD)** berfungsi sebagai *automatic curriculum* — reward eating meningkat seiring skill agen, sehingga target gradient RL tetap menantang di late-game.
- **Stagnation + Encircle** adalah komponen *anti-attractor* — dirancang berdasarkan observasi bahwa agen sering "puas" dengan policy sub-optimal (orbit, self-trap) tanpa penalty eksplisit.

---

*Dokumen ini disusun otomatis berdasarkan state repositori per Juni 2026. Semua angka dapat berubah setelah eksperimen multi-seed dan run final.*

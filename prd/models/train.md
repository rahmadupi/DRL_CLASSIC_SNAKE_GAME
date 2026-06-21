# TRAINING INFRASTRUCTURE & TUI

Modul pelatihan menggunakan pendekatan paralelisasi (_SubprocVecEnv_) dan dioperasikan melalui antarmuka _Terminal User Interface_ (TUI) berbasis `curses` di skrip peluncur utama (`train.py`).

## Spesifikasi Antarmuka Curses (State Machine)

TUI beroperasi sepenuhnya via _keyboard_, menampilkan 5 menu utama:

1. **[Algorithm]:** PPO vs DQN (Kiri/Kanan).
2. **[Existing Model]:** Navigasi vertikal membaca isi direktori `saved_models/*.zip`. Menyertakan opsi `[0] <New Model>`.
3. **[Output Prefix]:** Input teks murni untuk prefix (Preview: `[prefix]_[algo]_level[X].zip`). Didukung fitur _Auto-Increment_ jika nama _file_ mengalami duplikasi.
4. **[Level Select]:** Level 1 hingga 5.
5. **[Parallelization]:** Kontrol utilitas CPU _core_ (1 hingga 8).
6. **[Episode]:** Jumlah episode pelatihan.

## Mode Pelatihan & Resolusi Logika

Untuk mencegah _fatal exception_ pada _Display Server_ OS akibat _multiprocessing_:

- Jika **Parallelization == 1**: Lingkungan menggunakan `DummyVecEnv`. Fitur (Demo Mode: Visual Enabled) dapat dipicu. Kecepatan _step_ di- _hardcode_ untuk observasi.
- Jika **Parallelization > 1**: Sistem secara paksa masuk ke mode _Headless_ penuh menggunakan `SubprocVecEnv` (Kecepatan komputasi maksimum).

## Checkpointing & Monitoring

- Memanfaatkan `CheckpointCallback` untuk menyimpan bobot secara berkala (misal: setiap 50k iterasi), memungkinkan analisis visual _time-lapse_ via `play_game.py`.
- Metrik performa diekspor ke `logs/tb_logs/` secara _real-time_ untuk analisis di TensorBoard.

## UI

[Title]

[Options]
[Algorithm] PPO <-> DQN
[Existing Model] <New Model> / saved*models/\*.zip
[Output Prefix] [prefix]*[algo]\_level[X].zip
[Level Select] 1 <-> 5
[Parallelization] 1 <-> 8 Check available CPU cores
[Episode] 0

[Start Training] (Press Enter)

[Guide & Messages]

highlight focused option with inverse color. Use arrow keys to navigate and Enter to select.

for option if user press Enter: it will display the option list in the middle of the screen, user can navigate using arrow keys and select using Enter. The selected option will be highlighted.

---

## Implemented Training Loop

Bagian ini mendokumentasikan _training loop_ yang sudah diimplementasikan pada modul `game/train/`. Loop ini dipicu dari TUI launcher (`train_launcher.py`) dengan menekan Enter pada baris **Start Training**.

### Entry Point & Alur Pemanggilan

```
train_launcher.py  ──Enter on Start──▶  _launch_training()
                                         │
                                         ├─ validate model/algorithm match
                                         ├─ validate file existence
                                         ├─ translate episodes → timesteps
                                         │
                                         ├─ PPO:  train_ppo(PPOTrainingConfig)
                                         └─ DQN:  train_dqn(DQNTrainingConfig)
                                                   │
                                                   ├─ make_vec_env (Dummy/Subproc)
                                                   ├─ resolve_logger_dir (TensorBoard)
                                                   ├─ build_ppo / build_dqn
                                                   ├─ CheckpointCallback
                                                   ├─ model.learn(total_timesteps=...)
                                                   ├─ env.close()
                                                   └─ auto_naming() → model.save()
```

### Translasi Episode → Timestep

`SB3.learn()` hanya menerima _budget_ berbasis **timestep**, bukan episode. Launcher mengubah input `Episode` UI menjadi `total_timesteps` lewat heuristic:

```python
total_timesteps = max(2_048, cfg.episodes * 2_048) if cfg.episodes > 0 else 200_000
```

- `Episode == 0` → gunakan default `200_000` (lewati translasi).
- `Episode > 0` → alokasikan `2_048` langkah per episode sebagai _upper-bound_ kasar (kematian dini ≈ 10 langkah, _run_ panjang ≈ 400 langkah).

### Konfigurasi Trainer (`PPOTrainingConfig` & `DQNTrainingConfig`)

| Field                 | PPO                | DQN                     | Catatan                                                         |
| --------------------- | ------------------ | ----------------------- | --------------------------------------------------------------- |
| `level`               | ✅                 | ✅                      | Curriculum level 1–5                                            |
| `obs_type`            | `"spatiotemporal"` | `"12bit"`               | Flat vector vs. 4×20×20 tensor                                  |
| `n_envs`              | bebas (1..cpu)     | clamp ke `max_n_envs=4` | DQN di-cap karena _off-policy bias_                             |
| `total_timesteps`     | default `500_000`  | default `200_000`       | Budget langkah environment                                      |
| `learning_rate`       | `5e-4` (default)   | `5e-4` (default)        | Default `5e-4`; PRD membandingkan `1e-4` vs `5e-4`              |
| `use_linear_schedule` | ✅ `True`          | ✅ `True`               | Linear decay LR (PPO juga `clip_range`) — lihat §LR Schedule    |
| `lr_end_fraction`     | `0.0`              | `0.0`                   | LR akhir = `learning_rate × lr_end_fraction`                    |
| `clip_end_fraction`   | `0.0` (PPO)        | —                       | clip_range akhir = `clip_range × clip_end_fraction` (hanya PPO) |
| `checkpoint_freq`     | `50_000`           | `50_000`                | Steps antar checkpoint                                          |
| `load_path`           | ✅                 | ✅                      | Curriculum / resume                                             |

Field tambahan spesifik algoritma:

- **PPO**: `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=0.5`, `cnn_channels=32`, `d_model=64`, `n_heads=4`, `use_attention=True`.
- **DQN**: `buffer_size=100_000`, `learning_starts=1_000` (0 saat _resume_), `batch_size=64`, `gamma=0.99`, `tau=1.0`, `train_freq=4`, `target_update_interval=1_000`, `exploration_fraction=0.1`, `exploration_initial_eps=1.0`, `exploration_final_eps=0.05`, `hidden_dim=64`, `features_dim=64`.

### Langkah 1 — Vectorised Environment (`make_vec_env`)

```python
env = make_vec_env(level=level, obs_type=obs_type, n_envs=n_envs, seed=seed)
```

- `n_envs == 1` → `DummyVecEnv` (in-process; _renderer_ Pygame masih bisa di-attach).
- `n_envs > 1` → `SubprocVecEnv` (paksa _headless_; worker tidak punya akses ke display server parent).
- Setiap env di-_wrap_ lewat thunk agar bisa di-pickle oleh `SubprocVecEnv`.

### Langkah 2 — Logger Directory (`resolve_logger_dir`)

Membuat direktori TensorBoard dengan skema penamaan yang sama dengan model:

```
logs/tb_logs/<prefix>_<algo>_level<L>[_<n>]/
└── checkpoints/
```

Auto-increment suffix `_1`, `_2`, ... bila nama sudah dipakai.

### Langkah 3 — Model Construction (`build_ppo` / `build_dqn`)

- **New run**: instantiate `PPO(...)` atau `DQN(...)` dengan policy kwargs kustom (`SpatiotemporalExtractor` atau `DQN12BitExtractor`).
- **Resume**: panggil `PPO.load()` / `DQN.load()` dengan `custom_objects={"learning_rate": ...}` agar schedule LR dapat di-override per-stage curriculum. Setelah `load()`, kedua trainer men-override `model.lr_schedule` (PPO juga `model.clip_range`) lewat helper bersama `build_lr_schedule(...)` di `game/train/utility.py`. Lihat §LR Schedule di bawah untuk motivasi linear decay.

### LR & Clip-Range Schedule (Linear Decay)

Kedua algoritma (PPO dan DQN) memakai **linear decay** terhadap `learning_rate` — dan PPO juga terhadap `clip_range` — mengikuti recipe asli PPO (Schulman et al. 2017). Helper bersama [`build_lr_schedule(learning_rate, use_linear, end_fraction)`](game/train/utility.py) membungkus SB3 `get_linear_fn(...)` / `get_schedule_fn(...)` dan dipakai oleh `build_ppo()` maupun `build_dqn()`.

| Field                 | Default | Arti                                                                        |
| --------------------- | ------- | --------------------------------------------------------------------------- |
| `use_linear_schedule` | `True`  | `True` → linear decay; `False` → konstan                                    |
| `lr_end_fraction`     | `0.0`   | LR akhir = `learning_rate × lr_end_fraction` (`0.0` = 0)                    |
| `clip_end_fraction`   | `0.0`   | clip_range akhir = `clip_range × clip_end_fraction` (`0.0` = 0). Hanya PPO. |

**Mengapa linear decay (bukan konstan)?** Schedule konstan pada policy yang sudah matang membuat optimizer terus membuat update besar dan "mengetuk" policy keluar dari region bagus — gejala yang muncul sebagai osilasi reward di akhir pelatihan (sekitar 500k–700k step pada run PPO Level 1). Decay linear mengecilkan step size seiring policy konvergen sehingga osilasi mereda. Untuk kembali ke schedule konstan, set `use_linear_schedule=False` di config (atau turunkan `learning_rate` manual saat resume).

**Override saat resume.** SB3 menyimpan schedule sebagai callable pada `model.lr_schedule` (dan `model.clip_range` untuk PPO). `PPO.load()` / `DQN.load()` me-restore schedule dari checkpoint — agar curriculum run dapat mengubah LR per-stage, kedua builder secara eksplisit men-override schedule tersebut setelah `load()`.

**`reset_num_timesteps` pada resume.** Schedule linear mensyaratkan `progress_remaining` walk dari 1.0 → 0.0 selama run baru. `train_ppo()` dan `train_dqn()` memaksa `reset_num_timesteps=True` ketika `use_linear_schedule=True`; jika konstan, default `reset_num_timesteps=(load_path is None)` dipertahankan agar sumbu-X TensorBoard tetap kontinu lintas run.

### Langkah 4 — Checkpoint Callback

```python
CheckpointCallback(
    save_freq=max(1, config.checkpoint_freq // max(1, config.n_envs)),  # PPO
    save_freq=max(1, config.checkpoint_freq),                            # DQN
    save_path=str(checkpoint_dir),
    name_prefix=f"{prefix}_{algo}_level{L}",
    save_replay_buffer=False,  # PPO: on-policy, tidak perlu buffer
    save_replay_buffer=True,   # DQN: replay buffer ikut di-save agar resume valid
    save_vecnormalize=False,
)
```

Catatan:

- `save_freq` PPO di-bagi `n_envs` karena SB3 menghitung frekuensi per-step, bukan per-rollout.
- DQN menyimpan replay buffer; PPO tidak.

### Langkah 5 — `model.learn(total_timesteps=...)`

```python
model.learn(
    total_timesteps=config.total_timesteps,
    callback=checkpoint_cb,
    tb_log_name="ppo_run" | "dqn_run",
    reset_num_timesteps=(config.load_path is None),
    progress_bar=progress_bar,
)
```

- `reset_num_timesteps=False` saat _resume_ **dengan schedule konstan** agar counter langkah tidak kembali ke nol (TensorBoard log berkelanjutan). Jika `use_linear_schedule=True`, dipaksa `True` (lihat §LR Schedule).
- Loop ini **satu-satunya sumber kebenaran** untuk budgeting — tidak ada _outer_ loop episode-based. Episode hanya muncul sebagai terminasi natural di dalam `env.step()`.
- `try/finally` memastikan `env.close()` dipanggil agar worker `SubprocVecEnv` tidak menggantung.

### Langkah 6 — Final Save (`auto_naming` → `model.save`)

```python
final_path = auto_naming(prefix=output_prefix, algo=algo, level=level)
model.save(str(final_path))
```

`auto_naming` menghasilkan path unik dengan suffix auto-increment untuk mencegah overwrite:

```
saved_models/snake_ppo_level1.zip
saved_models/snake_ppo_level1_1.zip  # jika sudah ada
```

### Progress Bar (`RewardProgressBarCallback`)

Selain `CheckpointCallback`, kedua trainer memasang [`RewardProgressBarCallback`](game/train/utility.py) yang mengganti SB3 built-in `progress_bar` dengan `tqdm` headless-friendly. Callback membaca `model.ep_info_buffer` (di-populate oleh `Monitor` wrapper) setiap step.

Postfix keys:

| Key     | Sumber                                                        | Catatan                                                   |
| ------- | ------------------------------------------------------------- | --------------------------------------------------------- |
| `rew`   | rolling mean `ep["r"]` atas ~100 episode terakhir             | Reward rata-rata episode                                  |
| `len`   | rolling mean `ep["snake_length"]` atas episode yang punya key | **Panjang tubuh ular saat episode berakhir**              |
| `eps`   | total episode selesai                                         | Format `eps/total_eps` bila TUI menentukan target episode |
| `envs`  | `model.n_envs`                                                | Hanya tampil bila `n_envs > 1`                            |
| `graph` | Unicode sparkline `▁▂▃▄▅▆▇█` dari reward per-episode          | Auto-scaled ke min/max lokal                              |

`snake_length` dibaca dari info dict terminal env (lihat `prd/game/environment.md § Info Dict pada Terminal Step` — `env.step()` sekarang menyertakan `snake_length` pada hasil `terminated`/`truncated`). `Monitor` me-merge key tersebut ke `ep_info_buffer`. Episode lama (sebelum field ini ditambahkan) di-skip dari rata-rata sehingga tidak merusak nilai `len`.

Contoh bar:

```
PPO level1:  45%|████▌     | 90k/200k [02:13<02:39, 678 step/s] {rew=-0.82, len=6.4, eps=42, ▃▅▆▄▅▇▆▅▄▃}
```

**Mengapa `len` lebih informatif daripada reward?** Reward pada Snake noise-led (tiap episode bisa melonjak +10 atau anjlok -10 pada satu event). Panjang tubuh saat episode berakhir (= `INITIAL_SNAKE_LENGTH + jumlah_makanan_yang_dimakan`) memantulkan skill secara lebih halus dan cocok untuk memantau tren kenaikan lambat — lihat juga `prd/models/tensorboard.md § Sinyal TensorBoard` untuk metrik SB3 yang lebih diagnostik (`rollout/ep_len_mean`, `train/approx_kl`, `train/entropy_loss`).

### Perbedaan Utama PPO vs DQN

| Aspek                     | PPO                                            | DQN                                            |
| ------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| VecEnv                    | `SubprocVecEnv` (jika `n_envs>1`)              | `DummyVecEnv` (default) atau capped multi-env  |
| Obs type                  | `spatiotemporal` (4×20×20 tensor)              | `12bit` (flat 12-dim vector)                   |
| Checkpoint `save_freq`    | dibagi `n_envs`                                | absolute                                       |
| Replay buffer             | ❌ tidak di-save                               | ✅ di-save untuk resume                        |
| Resume LR                 | di-override lewat `build_lr_schedule` (linear) | di-override lewat `build_lr_schedule` (linear) |
| Default `total_timesteps` | `500_000`                                      | `200_000`                                      |
| TUI parallelization       | 1..cpu_count                                   | 1..4 (hard cap)                                |

### Algoritma Pemilihan Model (Resume Guard)

Sebelum `train_ppo` / `train_dqn` dipanggil, TUI launcher memvalidasi:

1. **Filename match** — pola `_(ppo|dqn)_level<digit>` harus cocok dengan algoritma aktif (lihat `_algo_from_filename`).
2. **File existence** — `Path(...).is_file()` dicek ulang sebelum loading (mencegah stale pick setelah file dihapus/dipindahkan).
3. **Learning starts** — untuk DQN `learning_starts=0` saat _resume_ (supaya tidak membuang langkah awal untuk random exploration ulang).

Ketiganya menghasilkan pesan error ramah dan kembali ke menu utama, alih-alih crash di dalam SB3 loader.

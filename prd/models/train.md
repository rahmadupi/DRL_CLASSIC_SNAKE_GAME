# TRAINING INFRASTRUCTURE & TUI

Modul pelatihan menggunakan pendekatan paralelisasi (_SubprocVecEnv_) dan dioperasikan melalui antarmuka _Terminal User Interface_ (TUI) berbasis `curses` di skrip peluncur utama (`train_launcher.py`).

## Spesifikasi Antarmuka Curses (State Machine)

TUI beroperasi sepenuhnya via _keyboard_, menampilkan **8 menu utama**:

1. **[Algorithm]:** PPO ↔ DQN (Kiri/Kanan).
2. **[Obs Type]:** _spatiotemporal_ ↔ _12bit_ (Kiri/Kanan). Tombol Enter membuka _modal picker_ dengan opsi ketiga `spatiotemporal_legacy` (backward-compat untuk model 4-channel lama). Nilai default berbeda per algoritma: PPO default `spatiotemporal`, DQN default `12bit`.
3. **[Existing Model]:** Navigasi vertikal membaca isi direktori `saved_models/*.zip`. Filter aktif: hanya tampilkan file yang **keduanya** cocok dengan `Algorithm` dan `Obs Type` aktif (regex di `_filter_models_for_algo` dan `_filter_models_for_obs_type`). Menyertakan opsi `[0] <New Model>`.
4. **[Output Prefix]:** Input teks murni untuk prefix (Preview: `[prefix]_[algo]_[obstype]_level[X].zip` — lihat §Skema Penamaan File). Didukung fitur _Auto-Increment_ jika nama _file_ mengalami duplikasi.
5. **[Level Select]:** Level 1 hingga 5.
6. **[Parallelization]:** Kontrol utilitas CPU _core_ (1 hingga 8 untuk PPO; di-cap ke 4 untuk DQN karena _off-policy bias_ — lihat §Algoritma Pemilihan Model).
7. **[Episode]:** Jumlah episode pelatihan. `0` artinya gunakan `total_timesteps` default dari JSON config.
8. **[Start Training]:** (Press Enter untuk memicu `_launch_training`).

> **Hubungan Antar Menu**: setiap perubahan pada `Algorithm` atau `Obs Type` memanggil `_drop_mismatched_model()`, yang akan mengosongkan `Existing Model` bila file yang dipilih sudah tidak kompatibel dengan combo baru. Hal yang sama terjadi jika user memilih model lewat picker lalu mengubah combo — picker akan menolak kombinasi yang tidak punya model tersimpan.

## Mode Pelatihan & Resolusi Logika

Untuk mencegah _fatal exception_ pada _Display Server_ OS akibat _multiprocessing_:

- Jika **Parallelization == 1**: Lingkungan menggunakan `DummyVecEnv`. Fitur (Demo Mode: Visual Enabled) dapat dipicu. Kecepatan _step_ di- _hardcode_ untuk observasi.
- Jika **Parallelization > 1**: Sistem secara paksa masuk ke mode _Headless_ penuh menggunakan `SubprocVecEnv` (Kecepatan komputasi maksimum).

## Checkpointing & Monitoring

- Memanfaatkan `CheckpointCallback` untuk menyimpan bobot secara berkala (misal: setiap 50k iterasi), memungkinkan analisis visual _time-lapse_ via `game_launcher.py`.
- Metrik performa diekspor ke `logs/tb_logs/` secara _real-time_ untuk analisis di TensorBoard. Direktori log mengikuti pola penamaan model dengan menyematkan token `obstype` — lihat `prd/models/tensorboard.md § Path TensorBoard` untuk detail.

## Skema Penamaan File (versi baru)

Model dan direktori log memakai pola:

```
saved_models/<prefix>_<algo>[_<obstype>]_level<L>[_<n>].zip
logs/tb_logs/<prefix>_<algo>[_<obstype>]_level<L>[_<n>]/
```

`<obstype>` adalah **token pendek** yang dipetakan dari nama panjang `obs_type`:

| `obs_type` (panjang)    | Token filename | Contoh model                                   |
| ----------------------- | -------------- | ---------------------------------------------- |
| `12bit`                 | `12bit`        | `snake_ppo_12bit_level1.zip`                   |
| `spatiotemporal`        | `sptmp`        | `snake_dqn_sptmp_level2_1.zip`                 |
| `spatiotemporal_legacy` | `sptmp_lgcy`   | `old_ppo_sptmp_lgcy_level1.zip` (4-channel v1) |

Pemetaan token ↔ nama panjang disimpan di `OBS_TYPE_TO_TOKEN` / `TOKEN_TO_OBS_TYPE` pada [`game/model/configs/__init__.py`](../../game/model/configs/__init__.py). Model lama **tanpa** token `<obstype>` (mis. `ext_ppo_level1.zip`, `mmm_dqn_level1.zip`) tetap dimuat — `game_launcher.py` mendeteksi `obs_type` lewat metadata SB3 (`observation_space.shape`) dengan fallback ke regex nama file bila shape lookup gagal. Lihat `game/env/input_controller.py::_infer_obs_type_from_filename` dan `_OBS_TYPE_FILENAME_PATTERNS` untuk regex pattern.

Regex nama file lengkap (`train_launcher.py::_ALGO_FROM_NAME`):

```regex
_(?P<algo>ppo|dqn)(?:_(?P<obstype>12bit|sptmp_lgcy|sptmp))?_level\d+
```

Group `obstype` bersifat _optional_ — model lama tanpa token cocok dengan pola ini karena `(?:_(?P<obstype>...))?` cocok dengan zero-length string.

## UI Mockup

```
        TRAIN SNAKE AGENT           device=cpu   cpus=8

  ▶[Algorithm]      PPO
   [Obs Type]       spatiotemporal (sptmp)
   [Existing Model] <New Model>
   [Output Prefix]  snake
   [Level Select]   1
   [Parallelization] 1 Demo mode
   [Episode]        0
   [Start Training]

  Preview → snake_ppo_sptmp_level1.zip
            /.../saved_models/snake_ppo_sptmp_level1.zip

       ↑/↓ row · ←/→ change · Enter select/edit · q quit
```

Saat kursor berpindah ke `Obs Type`, `Existing Model` langsung ter-reset bila model yang sedang dipilih tidak cocok dengan obs_type baru. Picker `Existing Model` otomatis memfilter sehingga hanya file yang **keduanya** cocok dengan `Algorithm` dan `Obs Type` yang ditampilkan.

---

## Konfigurasi Hyperparameter via JSON

Hyperparameter default disimpan di dua file JSON di `game/model/configs/`:

- **[`ppo_config.json`](../../game/model/configs/ppo_config.json)** — Default PPO (12bit dan spatiotemporal).
- **[`dqn_config.json`](../../game/model/configs/dqn_config.json)** — Default DQN (12bit dan spatiotemporal).

Kedua trainer memuat JSON sebagai basis lewat `PPOTrainingConfig.from_json_dict(...)` / `DQNTrainingConfig.from_json_dict(...)`. TUI launcher menambahkan override per-run (level, obs_type, total_timesteps, learning_rate, n_envs, dll.) lewat kwargs — kwargs yang match dengan field dataclass menang atas nilai JSON. Ini menjadikan JSON satu-satunya sumber kebenaran untuk "knob yang jarang diubah", sementara TUI men-override parameter per-run tanpa edit file.

Contoh panggilan:

```python
config = PPOTrainingConfig.from_json_dict(
    "ppo",                          # argumen pertama: "ppo" | "dqn" | path | dict
    level=3,                        # override
    obs_type="12bit",               # override
    learning_rate=2e-4,             # override
    total_timesteps=300_000,        # override
)
```

---

## Implemented Training Loop

Bagian ini mendokumentasikan _training loop_ yang sudah diimplementasikan pada modul `game/train/`. Loop ini dipicu dari TUI launcher (`train_launcher.py`) dengan menekan Enter pada baris **Start Training**.

### Entry Point & Alur Pemanggilan

```
train_launcher.py  ──Enter on Start──▶  _launch_training()
                                         │
                                         ├─ validate model/algorithm match
                                         ├─ validate model/obs_type match
                                         ├─ validate file existence
                                         ├─ translate episodes → timesteps
                                         │
                                         ├─ PPO:  PPOTrainingConfig.from_json_dict("ppo", ...)
                                         │         → train_ppo(config)
                                         │
                                         └─ DQN:  DQNTrainingConfig.from_json_dict("dqn", ...)
                                                   → train_dqn(config)
                                                   │
                                                   ├─ make_vec_env (Dummy/Subproc)
                                                   ├─ resolve_logger_dir (TensorBoard)
                                                   ├─ build_ppo / build_dqn
                                                   │     (pilih features extractor
                                                   │      berdasarkan obs_type)
                                                   ├─ CheckpointCallback
                                                   ├─ model.learn(total_timesteps=...)
                                                   ├─ env.close()
                                                   └─ auto_naming() → model.save()
```

Pemilihan features extractor di `build_ppo`/`build_dqn` adalah `if obs_type == "12bit": <MLP factory> else: <Spatiotemporal factory>` — lihat [game/train/ppo_trainer.py](../../game/train/ppo_trainer.py) dan [game/train/dqn_trainer.py](../../game/train/dqn_trainer.py) untuk detail lengkap.

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
| `obs_type`            | `"spatiotemporal"` | `"12bit"`               | Bisa juga `"spatiotemporal_legacy"` (4-channel v1)              |
| `n_envs`              | bebas (1..cpu)     | clamp ke `max_n_envs=4` | DQN di-cap karena _off-policy bias_                             |
| `total_timesteps`     | default `500_000`  | default `200_000`       | Budget langkah environment                                      |
| `learning_rate`       | `1e-3` (default)   | `5e-4` (default)        | Default di JSON; PRD membandingkan `1e-4` vs `5e-4`             |
| `use_linear_schedule` | ✅ `True`          | ✅ `True`               | Linear decay LR (PPO juga `clip_range`) — lihat §LR Schedule    |
| `lr_end_fraction`     | `0.1`              | `0.0`                   | LR akhir = `learning_rate × lr_end_fraction`                    |
| `clip_end_fraction`   | `0.05` (PPO)       | —                       | clip_range akhir = `clip_range × clip_end_fraction` (hanya PPO) |
| `checkpoint_freq`     | `50_000`           | `50_000`                | Steps antar checkpoint                                          |
| `load_path`           | ✅                 | ✅                      | Curriculum / resume                                             |

Field tambahan spesifik algoritma (saat ini nilainya di JSON):

- **PPO**: `n_steps=2048`, `batch_size=256`, `n_epochs=4`, `gae_lambda=0.95`, `clip_range=0.2`, `clip_range_vf=0.2`, `ent_coef=0.01`, `vf_coef=0.4`, `max_grad_norm=0.5`, `cnn_channels=32`, `d_model=64`, `n_heads=4`, `dropout=0.1`, `use_attention=True`, `net_arch_pi=[64]`, `net_arch_vf=[64]`.
- **DQN**: `buffer_size=100_000`, `learning_starts=1_000` (0 saat _resume_), `batch_size=64`, `gamma=0.99`, `tau=1.0`, `train_freq=4`, `gradient_steps=1`, `target_update_interval=1_000`, `exploration_fraction=0.1`, `exploration_initial_eps=1.0`, `exploration_final_eps=0.05`, `hidden_dim=64`, `features_dim=64`, `cnn_channels=32`, `d_model=64`, `n_heads=8`, `use_attention=True`.

> **`clip_range_vf`** adalah trust-region clip pada value head (critic), bukan policy. Sama seperti `clip_range` mencegah policy drift, `clip_range_vf` mencegah value head dari "diyakin-yakinkan" oleh outlier return dalam satu update. Tanpa itu, value function bisa diverge (`explained_variance` turun, `value_loss` naik) — yang kemudian membuat GAE advantages jadi noisi dan policy gradient tidak stabil. Default 0.2 di config PPO.

> **Cara edit**: buka JSON di `game/model/configs/`, ubah nilai, simpan. Tidak perlu restart apa-apa — JSON dibaca ulang tiap kali `_from_json_dict()` dipanggil (yaitu di awal setiap run).

### Langkah 1 — Vectorised Environment (`make_vec_env`)

```python
env = make_vec_env(level=level, obs_type=obs_type, n_envs=n_envs, seed=seed)
```

- `n_envs == 1` → `DummyVecEnv` (in-process; _renderer_ Pygame masih bisa di-attach).
- `n_envs > 1` → `SubprocVecEnv` (paksa _headless_; worker tidak punya akses ke display server parent).
- Setiap env di-_wrap_ lewat thunk agar bisa di-pickle oleh `SubprocVecEnv`.

### Langkah 2 — Logger Directory (`resolve_logger_dir`)

Membuat direktori TensorBoard dengan skema penamaan yang sama dengan model (termasuk token `<obstype>`):

```
logs/tb_logs/<prefix>_<algo>[_<obstype>]_level<L>[_<n>]/
└── checkpoints/
```

Auto-increment suffix `_1`, `_2`, ... bila nama sudah dipakai. Contoh konkret:

```
logs/tb_logs/snake_ppo_sptmp_level1/
logs/tb_logs/snake_ppo_12bit_level1/
logs/tb_logs/snake_dqn_sptmp_level1/
logs/tb_logs/snake_dqn_12bit_level1/
```

Empat direktori di atas adalah run terpisah di TensorBoard — bisa dibandingkan side-by-side untuk _cross-comparison_ architecture × algorithm.

### Langkah 3 — Model Construction (`build_ppo` / `build_dqn`)

- **New run**: instantiate `PPO(...)` atau `DQN(...)` dengan policy kwargs kustom:
  - `obs_type == "12bit"` → `make_ppo_policy_kwargs(...)` / `make_dqn_policy_kwargs(...)` dengan MLP features extractor (3-layer Dense).
  - `obs_type ∈ {"spatiotemporal", "spatiotemporal_legacy"}` → factory CNN+Attention yang membaca channel count dari `observation_space.shape[0]` (saat ini selalu 4-channel honest layout).
- **Resume**: panggil `PPO.load()` / `DQN.load()` dengan `custom_objects={"learning_rate": ...}` agar schedule LR dapat di-override per-stage curriculum. Setelah `load()`, kedua trainer men-override `model.lr_schedule` (PPO juga `model.clip_range`) lewat helper bersama `build_lr_schedule(...)` di `game/train/utility.py`. Lihat §LR Schedule di bawah untuk motivasi linear decay.

### LR & Clip-Range Schedule

PPO dan DQN mendukung dua mode schedule lewat `use_linear_schedule`:

- **`use_linear_schedule=true`** → linear decay dari `learning_rate` ke `learning_rate × lr_end_fraction` selama run. PPO juga men-decay `clip_range` ke `clip_range × clip_end_fraction` (terkontrol via `clip_end_fraction`).
- **`use_linear_schedule=false`** → konstan selama run (schedule_fn dari SB3).

Helper bersama [`build_lr_schedule(learning_rate, use_linear, end_fraction)`](../../game/train/utility.py) membungkus SB3 `get_linear_fn(...)` / `get_schedule_fn(...)` dan dipakai oleh `build_ppo()` maupun `build_dqn()`. PPO punya helper setara [`_build_clip_schedule`](../../game/train/ppo_trainer.py).

| Field                 | Default (saat ini) | Arti                                                                     |
| --------------------- | ------------------ | ------------------------------------------------------------------------ |
| `use_linear_schedule` | `true`             | `true` → linear decay; `false` → konstan sepanjang run                   |
| `lr_end_fraction`     | `0.1`              | LR akhir = `learning_rate × lr_end_fraction` (hanya relevan jika linear) |
| `clip_end_fraction`   | `1.0`              | clip_range akhir = `clip_range × clip_end_fraction`. Hanya PPO.          |

**Default config saat ini**: `learning_rate=5e-5`, `use_linear_schedule=true`, `lr_end_fraction=0.1`, `clip_end_fraction=1.0`. Artinya:

- LR decays 5e-5 → 5e-6 selama 500k steps (linear)
- clip_range **konstan** di 0.2 selama run

**Mengapa LR masih decay, tapi clip_range konstan?** Snake punya tiga karakteristik yang membuat linear decay LR saja problematik:

1. **Sparse reward** — event `+10`/`-10` sangat jarang (~300 langkah per makanan). Gradient signal yang sampai ke optimizer sudah noisy; LR decay mengecilkan langkah di akhir run, membantu **fine-tune** tanpa overshooting.
2. **Small network (~75k params)** — model kecil mudah "overshoot" region bagus dalam satu update besar. Decay LR memberikan margin.
3. **Late-game fine-tuning** — setelah policy menemukan region bagus, decay membantu settle tanpa fluktuasi besar.

Tapi **`clip_range` decay harus dihindari**. Run PPO Level 1 sebelumnya (LR 1e-4 → 1e-5, clip_range 0.2 → 0.01) menunjukkan pola klasik: **peak performance** di iter ~24 (`eating_rate=0.98`), lalu **crash** di iter 27 (`eating_rate=0.41`) ketika `clip_range` sudah cukup kecil (0.115) untuk "menjepit" policy ke update yang noisy. `clip_end_fraction=1.0` memutus pola ini dengan menjaga trust region tetap longgar sepanjang run.

**Kapan linear decay penuh (LR + clip_range) masih berguna?** Untuk **fine-tuning** model yang sudah konvergen, di mana policy benar-benar stabil. Workflow dua tahap: (1) train dengan LR-decay + clip_range-konstan sampai converged, (2) resume dengan `clip_end_fraction` lebih kecil (mis. 0.5) untuk fine-tune.

**Override saat resume.** SB3 menyimpan schedule sebagai callable pada `model.lr_schedule` (dan `model.clip_range` untuk PPO). `PPO.load()` / `DQN.load()` me-restore schedule dari checkpoint — agar curriculum run dapat mengubah LR per-stage, kedua builder secara eksplisit men-override schedule tersebut setelah `load()`. `clip_range_vf` juga di-restore via `custom_objects` agar resumed run memakai nilai baru.

**`reset_num_timesteps` pada resume.** Schedule linear mensyaratkan `progress_remaining` walk dari 1.0 → 0.0 selama run baru. `train_ppo()` dan `train_dqn()` memaksa `reset_num_timesteps=True` ketika `use_linear_schedule=True`; jika konstan, default `reset_num_timesteps=(load_path is None)` dipertahankan agar sumbu-X TensorBoard tetap kontinu lintas run.

### Trust-Region pada Value Head (`clip_range_vf`)

PPO memakai `clip_range` untuk menjaga policy update dalam trust region di sekitar policy lama. Tapi **value head (critic)** di-update terpisah, secara default dengan MSE loss tanpa clipping. Pada Snake, return episode sangat bervariasi (long survival → return besar; quick death → return kecil), dan outlier ini bisa "menyentil" value head dalam satu update.

`clip_range_vf` (parameter SB3 PPO, default None = tidak ada clipping) menerapkan trust region yang sama pada value update:

```
values_clipped = value_old + clip(value_new − value_old, −clip_range_vf, +clip_range_vf)
value_loss = max(MSE(value_new, returns), MSE(values_clipped, returns))
```

Default `clip_range_vf=0.2` di `ppo_config.json`. Indikator bahwa ia bekerja:

| Signal               | Tanpa `clip_range_vf`    | Dengan `clip_range_vf=0.2`       |
| -------------------- | ------------------------ | -------------------------------- |
| `value_loss`         | Bisa spike / climb       | Stabil, gradual changes          |
| `explained_variance` | Drop terus (0.5 → 0.3)   | Stabil di atas 0.4 sepanjang run |
| `clip_fraction`      | Naik (policy divergence) | Stabil / turun                   |

Feedback loop positif tanpa `clip_range_vf`: value function diverge → GAE advantages noisi → policy gradient noisi → value function diverge lagi. Mengaktifkan `clip_range_vf` memutus loop ini sejak awal.

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
    reset_num_timesteps=(config.load_path is None) or config.use_linear_schedule,
    progress_bar=progress_bar,
)
```

- `reset_num_timesteps=False` saat _resume_ **dengan schedule konstan** agar counter langkah tidak kembali ke nol (TensorBoard log berkelanjutan). Jika `use_linear_schedule=True`, dipaksa `True` (lihat §LR Schedule).
- Loop ini **satu-satunya sumber kebenaran** untuk budgeting — tidak ada _outer_ loop episode-based. Episode hanya muncul sebagai terminasi natural di dalam `env.step()`.
- `try/finally` memastikan `env.close()` dipanggil agar worker `SubprocVecEnv` tidak menggantung.

### Langkah 6 — Final Save (`auto_naming` → `model.save`)

```python
final_path = auto_naming(prefix=output_prefix, algo=algo, level=level, obs_type=obs_type)
model.save(str(final_path))
```

`auto_naming` menghasilkan path unik dengan suffix auto-increment untuk mencegah overwrite:

```
saved_models/snake_ppo_sptmp_level1.zip
saved_models/snake_ppo_sptmp_level1_1.zip  # jika sudah ada
```

Lihat §Skema Penamaan File untuk token `<obstype>`.

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
PPO spatiotemporal level1:  45%|████▌     | 90k/200k [02:13<02:39, 678 step/s] {rew=-0.82, len=6.4, eps=42, ▃▅▆▄▅▇▆▅▄▃}
```

Deskripsi bar sekarang menyertakan `obs_type` agar user tahu pipeline mana yang sedang berjalan.

**Mengapa `len` lebih informatif daripada reward?** Reward pada Snake noise-led (tiap episode bisa melonjak +10 atau anjlok -10 pada satu event). Panjang tubuh saat episode berakhir (= `INITIAL_SNAKE_LENGTH + jumlah_makanan_yang_dimakan`) memantulkan skill secara lebih halus dan cocok untuk memantau tren kenaikan lambat — lihat juga `prd/models/tensorboard.md § Sinyal TensorBoard` untuk metrik SB3 yang lebih diagnostik (`rollout/ep_len_mean`, `train/approx_kl`, `train/entropy_loss`).

### Spawn-Proximity Curriculum (Warm-up)

`game_environment` mendukung kurikulum penempatan target agar pelatihan tidak dimulai dari konfigurasi tersulit. Saat `SPAWN_PROXIMITY_ENABLED = true` di `game/env/config.json`, _initial food_ ditempatkan di dalam bola Manhattan radius `SPAWN_PROXIMITY_RADIUS` sel di sekitar kepala ular, dan hanya selama `SPAWN_PROXIMITY_STEPS` _env steps_ pertama. Tujuannya: agar agen mengenali asosiasi "makan → reward positif" sebelum harus menjelajah grid 20×20 secara membabi buta.

Detail lengkap (filter jarak, fallback, perilaku _respawn_ setelah makan) didokumentasikan di [`prd/game/food.md § Spawn-Proximity Curriculum`](../game/food.md). Catatan yang relevan untuk pipeline pelatihan:

- **Counter adalah per-proses.** Di `SubprocVecEnv`, setiap _worker_ mempertahankan kontributor sendiri, sehingga jendela _warm-up_ dihitung per env-step, bukan total langkah absolut. Untuk rollout paralel, kalikan `SPAWN_PROXIMITY_STEPS` dengan `n_envs` agar _warm-up_ sesuai dengan total langkah yang direncanakan.
- **Hanya penempatan awal episode.** Episode yang dimulai setelah jendela terlampaui (atau setelah fitur dimatikan) kembali memakai sampler acak global.
- **Override melalui config saja.** Tidak ada argumen runtime — aktif/nonaktifkan dilakukan sebelum `train_ppo()` / `train_dqn()` dipanggil dengan mengubah `config.json` (atau via _custom_ JSON loader bila Anda menambahkan wrapper).

Disarankan sebagai _default_ untuk run PPO/DQN Level 1 di mana agen mulai dari bobot acak: aktifkan _warm-up_ untuk ~50k env-steps pertama, lalu biarkan proximity _off_ selama sisa pelatihan. Untuk level lanjut (≥ 3) di mana agen me-_resume_ dari bobot level-1, biarkan _off_ karena kemampuan navigasi sudah ada.

### Perbedaan Utama PPO vs DQN

| Aspek                     | PPO                                            | DQN                                            |
| ------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| VecEnv                    | `SubprocVecEnv` (jika `n_envs>1`)              | `DummyVecEnv` (default) atau capped multi-env  |
| Obs type (default)        | `spatiotemporal` (4×20×20 honest tensor)       | `12bit` (flat 12-dim vector)                   |
| Obs type (alternatif)     | `12bit` atau `spatiotemporal_legacy`           | `spatiotemporal` atau `spatiotemporal_legacy`  |
| Checkpoint `save_freq`    | dibagi `n_envs`                                | absolute                                       |
| Replay buffer             | ❌ tidak di-save                               | ✅ di-save untuk resume                        |
| Resume LR                 | di-override lewat `build_lr_schedule` (linear) | di-override lewat `build_lr_schedule` (linear) |
| Default `total_timesteps` | `500_000`                                      | `200_000`                                      |
| TUI parallelization       | 1..cpu_count                                   | 1..4 (hard cap)                                |
| Default `learning_rate`   | `1e-3`                                         | `5e-4`                                         |

### Algoritma Pemilihan Model (Resume Guard)

Sebelum `train_ppo` / `train_dqn` dipanggil, TUI launcher memvalidasi tiga hal:

1. **Filename — algorithm match** — pola `_(algo)[_(obstype)]?_level<digit>` harus cocok dengan `Algorithm` aktif (lihat `_algo_from_filename`). Model lama tanpa token obs_type lolos validasi ini; model baru harus punya token yang cocok.
2. **Filename — obs_type match** — kalau nama file punya token `<obstype>`, token itu harus sama dengan `Obs Type` aktif (lihat `_obs_type_from_filename` dan `_drop_mismatched_model`). File tanpa token lolos (fallback ke metadata SB3 saat load).
3. **File existence** — `Path(...).is_file()` dicek ulang sebelum loading (mencegah stale pick setelah file dihapus/dipindahkan).

Untuk DQN, tambahan kecil: `learning_starts=0` saat _resume_ supaya tidak membuang langkah awal untuk random exploration ulang.

Ketiganya menghasilkan pesan error ramah dan kembali ke menu utama, alih-alih crash di dalam SB3 loader. Picker di TUI sudah pre-filter dengan logika yang sama, jadi kombinasi yang sampai ke validasi akhir biasanya hanya muncul karena rename manual atau corruption.

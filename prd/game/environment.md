# ENVIRONMENT & STATE SPACE

Lingkungan beroperasi menggunakan infrastruktur _Dual-Outlet State Space_ (ditambah _legacy_ untuk backward-compat) untuk memfasilitasi eksperimen komparatif. Kelas `game_environment` menerima parameter `obs_type` untuk menentukan _output_ matriks.

## Ringkasan Outlet

| `obs_type`                | Shape         | Digunakan oleh     | Status           |
| ------------------------- | ------------- | ------------------ | ---------------- |
| `"spatiotemporal"`        | `(8, 20, 20)` | PPO (default), DQN | **Default baru** |
| `"spatiotemporal_legacy"` | `(4, 20, 20)` | PPO/DQN            | Backward-compat  |
| `"12bit"`                 | `(12,)`       | DQN (default), PPO | Sesuai paper     |

Pemilihan outlet sekarang bisa dilakukan dari TUI launcher lewat baris `[Obs Type]` (lihat `prd/models/train.md § Spesifikasi Antarmuka Curses`). Pada akhirnya, empat varian arsitektur × algoritma semuanya membaca dari outlet yang sama; hanya _features extractor_ di `game/model/` yang berbeda.

## 1. Spatiotemporal Outlet (v2, default)

Mengekspor Tensor 3D berukuran **8×20×20** yang memetakan relasi geometris seketika dan _high-level_ cues:

- **Ch0 Wall:** Batas dinding absolut (Biner 0.0 atau 1.0).
- **Ch1 Decaying Body:** Tubuh ular. Kepala = 1.0, nilai terdegradasi secara linear berdasarkan indeks segmen hingga mendekati 0.0 di ekor. Berfungsi sebagai heuristik anti-_Greedy Trap_.
- **Ch2 Static Food:** Koordinat makanan statis (1.0).
- **Ch3 Dynamic Food:** Peta per-sel posisi makanan dinamis. Posisi saat ini $t$ = 1.0, posisi sebelumnya $t-1$ = 0.5. Sinyal ini krusial untuk Level 3-4 di mana makanan dinamis adalah satu-satunya target — tanpa Ch3, agen hanya punya directional signal (Ch5) tanpa info posisi jangka panjang.
- **Ch4 Head Direction:** 1.0 di sel yang akan dimasuki kepala pada langkah berikutnya (1 sel searah `direction_idx`).
- **Ch5 Food Direction:** 1.0 di sel yang berdekatan dengan kepala pada salah satu dari 4 arah absolut (UP/RIGHT/DOWN/LEFT) di mana makanan ada; mirror Bits 8-11 di outlet 12-bit.
- **Ch6 Relative Danger:** 1.0 di sel STRAIGHT/LEFT/RIGHT (relatif terhadap heading kepala) yang berbahaya menurut `DANGER_FROM_*` flags di config (wall / body / tail-exception). Sel "behind" diomit karena 180° reversal tidak mungkin.
- **Ch7 Snake Length:** Broadcast `len(snake) / 400` ke semua sel — sinyal global untuk policy memantau pertumbuhan.

Channel Ch0-Ch3 identik dengan outlet v1 di bawah; Ch4-Ch7 adalah _additive_ untuk menangkap sinyal-sinyal yang hilang di v1 (head dir, food dir, danger relatif, panjang tubuh).

## 2. Spatiotemporal Outlet (v1, legacy)

Backward-compat outlet dengan **4×20×20** tensor. Ch0 Wall, Ch1 Decaying Body, Ch2 Static Food, Ch3 Dynamic Momentum. Dipakai untuk memuat model lama yang dilatih sebelum Ch4-Ch7 ditambahkan; **tidak disarankan untuk training baru** — gunakan `obs_type="spatiotemporal"` (v2) sebagai gantinya.

## 3. 12-bit Baseline Outlet

Mengekspor _array_ 1D berisi 12 angka biner untuk mereplikasi _environment baseline_ sesuai spesifikasi literatur (menyatakan keberadaan rintangan dan arah makanan relatif terhadap kepala):

- **Bits 0-3:** Obstacle (wall/body) di arah [UP, RIGHT, DOWN, LEFT].
- **Bits 4-7:** Body proximity (1 langkah ke depan).
- **Bits 8-11:** Relative food direction (signs: dx > 0, dx < 0, dy > 0, dy < 0).

Sesuai paper "Deep Q-Snake"; dipakai oleh baseline `dqn_12bit.py` dan sebagai state representation kedua untuk eksperimen PPO 12-bit (`ppo_12bit.py`).

## Sistem Level (Level 1-5)

Lingkungan menerima parameter `level` untuk mengatur kompleksitas inisialisasi:

- **Level 1:** 1 Target Statis murni.
- **Level 2:** Multi-Target Statis.
- **Level 3:** 1 Target Dinamis (_Stochastic Momentum_).
- **Level 4:** Multi-Target Dinamis.
- **Level 5:** 3 Static Food + 2 Dynamic Food (Statis + Dinamis secara bersamaan).

## Konfigurasi Peta

- **Dimensi:** Grid 20x20 blok.

## Info Dict pada Terminal Step

`env.step()` mengembalikan `info` dict yang sekarang menyertakan `snake_length` setiap kali episode berakhir (`terminated` atau `truncated`). Nilai `snake_length` adalah panjang tubuh ular saat step tersebut - sama dengan `INITIAL_SNAKE_LENGTH + jumlah_makanan_yang_dimakan`.

| Termination reason | Sumber `snake_length`                                                |
| ------------------ | -------------------------------------------------------------------- |
| `"collision"`      | `len(self.snake)` **sebelum** head baru di-append                    |
| `"win"`            | `len(self.snake)` setelah append (= `MAX_GRID_AREA` = 400)           |
| `"truncated"`      | `len(self.snake)` setelah append pada step yang melebihi `max_steps` |

`Monitor` wrapper (di-install oleh `make_vec_env` di `game/train/utility.py`) me-merge field ini ke `ep_info_buffer`, sehingga tersedia untuk [`RewardProgressBarCallback`](game/train/utility.py) (postfix `len=`) dan analisis pasca-training. Episode lama yang tidak memiliki key `snake_length` di-skip dari rata-rata rolling window.

## Deteksi `obs_type` Saat Loading Model

`game_launcher.py` saat memuat model `.zip` mendeteksi `obs_type` lewat dua lapis fallback:

1. **SB3 metadata** (`observation_space.shape`) — diparse dari field `data` JSON di dalam zip via `_read_sb3_metadata` di `game/env/input_controller.py`. Mapping shape → obs_type:

   | Shape         | `obs_type`              |
   | ------------- | ----------------------- |
   | `(8, 20, 20)` | `spatiotemporal`        |
   | `(4, 20, 20)` | `spatiotemporal_legacy` |
   | `(12,)`       | `12bit`                 |

2. **Regex nama file** (fallback) — bila metadata tidak bisa di-decode (pickle gagal, format lama), pola `_OBS_TYPE_FILENAME_PATTERNS` di `input_controller.py` mencoba mencocokkan token `12bit`, `sptmp`, atau `sptmp_lgcy` di nama file. Token panjang `spatiotemporal` (tanpa suffix `_lgcy`) juga diterima sebagai alias.

Setelah `obs_type` terdeteksi, `game_environment` dibangun dengan parameter tersebut sehingga `observation_space` selalu cocok dengan policy yang sudah dilatih — `model.predict()` tidak akan crash karena shape mismatch.

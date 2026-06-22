# ENVIRONMENT & STATE SPACE

Lingkungan beroperasi menggunakan infrastruktur _Dual-Outlet State Space_ (ditambah _legacy_ untuk backward-compat) untuk memfasilitasi eksperimen komparatif. Kelas `game_environment` menerima parameter `obs_type` untuk menentukan _output_ matriks.

## Ringkasan Outlet

| `obs_type`                | Shape         | Digunakan oleh     | Status                    |
| ------------------------- | ------------- | ------------------ | ------------------------- |
| `"spatiotemporal"`        | `(4, 20, 20)` | PPO (default), DQN | **Default — honest layout** |
| `"spatiotemporal_legacy"` | `(4, 20, 20)` | PPO/DQN            | Alias (backward-compat)   |
| `"12bit"`                 | `(12,)`       | DQN (default), PPO | Sesuai paper              |

Pemilihan outlet sekarang bisa dilakukan dari TUI launcher lewat baris `[Obs Type]` (lihat `prd/models/train.md § Spesifikasi Antarmuka Curses`). Pada akhirnya, empat varian arsitektur × algoritma semuanya membaca dari outlet yang sama; hanya _features extractor_ di `game/model/` yang berbeda.

## 1. Spatiotemporal Outlet (honest layout, default)

Mengekspor Tensor 3D berukuran **4×20×20** yang hanya memuat relasi geometrik mentah — tanpa _heuristic crutches_:

- **Ch0 Wall:** Batas dinding absolut (Biner 0.0 atau 1.0).
- **Ch1 Decaying Body:** Tubuh ular. Kepala = 1.0, nilai terdegradasi secara linear berdasarkan indeks segmen hingga mendekati 0.0 di ekor. Berfungsi sebagai heuristik anti-_Greedy Trap_ dan secara implisit menyandi panjang tubuh lewat _extent_-nya.
- **Ch2 Static Food:** Koordinat makanan statis (1.0).
- **Ch3 Dynamic Food:** Peta per-sel posisi makanan dinamis. Posisi saat ini $t$ = 1.0, posisi sebelumnya $t-1$ = 0.5. Sinyal ini krusial untuk Level 3-4 di mana makanan dinamis adalah satu-satunya target — momentum $t-1$ adalah satu-satunya cara agen mengetahui ke arah mana makanan bergerak.

### Apa yang TIDAK ada di sini (alasan _honest_)

Versi v2 sebelumnya menambahkan 4 channel lagi di atas Ch0-Ch3 (Ch4 Head Direction, Ch5 Food Direction, Ch6 Relative Danger, Ch7 broadcast Snake Length) — semua itu adalah _heuristic crutches_ yang membuat jaringan membaca jawabannya secara langsung alih-alih mempelajarinya dari geometri:

- **Head direction** — dapat dideduksi dari gradien degradasi Ch1.
- **Food direction** — dapat dideduksi dari momentum Ch3 + posisi Ch2.
- **Relative danger** — dapat dideduksi dari Ch0 (walls) + Ch1 (body).
- **Snake length** — tidak lagi di-broadcast; agen harus belajar dari extent Ch1 atau sinyal rollout implisit.

Pengangkatan channel-channel ini adalah apa yang membuat perbandingan PPO↔DQN dan 12-bit↔spatiotemporal benar-benar mengukur algoritma, bukan _cheat-sheet_.

## 2. Spatiotemporal Legacy Outlet

Alias untuk outlet default di atas — **4×20×20** tensor dengan layout identik. Disimpan murni agar model lama dengan token `sptmp_lgcy` di nama file tetap ter-resolve lewat regex deteksi obs_type. Tidak ada bedanya dengan `spatiotemporal` di sisi runtime.

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
   | `(4, 20, 20)` | `spatiotemporal`        |
   | `(12,)`       | `12bit`                 |

   > Catatan: shape `(4, 20, 20)` sekarang cocok dengan **kedua** `spatiotemporal` dan `spatiotemporal_legacy` (kedua obs_type menghasilkan tensor identik). Regex nama file adalah fallback utama untuk membedakan keduanya; jika metadata SB3 tidak bisa di-decode, nama file menjadi penentu.

2. **Regex nama file** (fallback) — bila metadata tidak bisa di-decode (pickle gagal, format lama), pola `_OBS_TYPE_FILENAME_PATTERNS` di `input_controller.py` mencoba mencocokkan token `12bit`, `sptmp`, atau `sptmp_lgcy` di nama file. Token panjang `spatiotemporal` (tanpa suffix `_lgcy`) juga diterima sebagai alias.

Setelah `obs_type` terdeteksi, `game_environment` dibangun dengan parameter tersebut sehingga `observation_space` selalu cocok dengan policy yang sudah dilatih — `model.predict()` tidak akan crash karena shape mismatch.

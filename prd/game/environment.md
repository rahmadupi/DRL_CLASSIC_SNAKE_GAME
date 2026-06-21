# ENVIRONMENT & STATE SPACE

Lingkungan beroperasi menggunakan infrastruktur _Dual-Outlet State Space_ untuk memfasilitasi eksperimen komparatif. Kelas `AdvancedSnakeEnv` menerima parameter `obs_type` untuk menentukan _output_ matriks.

## 1. Spatiotemporal Outlet (Untuk PPO)

Mengekspor Tensor 3D berukuran 20x20x4 yang memetakan relasi geometris seketika:

- **Channel 1 (Obstacle):** Batas dinding absolut (Biner 0.0 atau 1.0).
- **Channel 2 (Decaying Body):** Tubuh ular. Kepala = 1.0, nilai terdegradasi secara linear berdasarkan indeks segmen hingga mendekati 0.0 di ekor. Berfungsi sebagai heuristik anti-_Greedy Trap_.
- **Channel 3 (Static Target):** Koordinat makanan statis (1.0).
- **Channel 4 (Dynamic Momentum):** Koordinat makanan dinamis yang merekam jejak vektor. Posisi $t$ = 1.0, posisi $t-1$ = 0.5.

## 2. 12-bit Baseline Outlet (Untuk DQN Paper)

Mengekspor _array_ 1D berisi 12 angka biner untuk mereplikasi _environment baseline_ sesuai spesifikasi literatur (menyatakan keberadaan rintangan dan arah makanan relatif terhadap kepala).

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

# TARGET ENTITIES BEHAVIOR

Target (makanan) dalam lingkungan dibagi menjadi dua varian dengan mekanika independen.

## 1. Target Statis

Entitas pasif yang dipijak di koordinat acak yang kosong. Terekam murni di Kanal 3.

## 2. Target Dinamis (Stochastic Momentum & Evasion)

Terekam di Kanal 4. Bergerak menggunakan _State Machine_ probabilistik untuk mensimulasikan mangsa hidup dengan efisiensi komputasi maksimal $O(1)$.

- **State 1 (Momentum Jarak):** Target memilih jarak tempuh acak (3 hingga 8 blok) dan melaju lurus sejauh jarak tersebut tanpa henti.
- **State 2 (Rotasi & Evasion):** Setelah limit jarak tercapai, target memutuskan arah baru berdasarkan probabilitas terbobot:
  - _Standard Rotation:_ Mengacak 4 arah (Atas, Bawah, Kiri, Kanan).
  - _Active Evasion (Probabilitas Rendah):_ Fungsi _Greedy Evasion_ dipanggil. Target mengevaluasi 4 kotak di sebelahnya secara instan dan memilih kotak dengan jarak _Euclidean_ terjauh dari kepala ular (_no future pathfinding_).
    probability weighted = 0.8 for Standard Rotation, 0.2 for Active Evasion.

## Proximity Collision Override

Jika target dinamis berjarak tepat 1 blok dari dinding atau target lain di jalur lurusnya (saat berada di State 1), ia wajib menghentikan siklus State 1 secara prematur dan langsung memicu State 2 pada _step_ berikutnya untuk mencegah tumpang tindih data tensor.

## 3. Spawn-Proximity Curriculum (Warm-up)

Saat fitur diaktifkan (`SPAWN_PROXIMITY_ENABLED = true` di `game/env/config.json`), penempatan awal target di setiap _episode_ dibatasi pada **bola Manhattan** dengan radius `SPAWN_PROXIMITY_RADIUS` sel di sekitar kepala ular — bukan dari seluruh grid. Tujuannya: mengajarkan asosiasi "makanan → reward positif" kepada agen di awal pelatihan ketika ia belum mampu menjelajah grid 20×20 secara efisien.

- **Aktif hanya untuk placement awal episode (`_init_food()`).** _Respawn_ setelah ular makan tetap memakai sampler acak global, sehingga perilaku _long-tail_ tidak berubah setelah _warm-up_ berakhir.
- **Jendela _warm-up_:** `SPAWN_PROXIMITY_STEPS` _env steps_ (kontributor level-kelas `game_environment._proximity_steps_taken`, dinaikkan sekali per `step()`). Setelah anggaran terlampaui, penempatan kembali ke perilaku acak standar.
- **Counter adalah per-proses Python.** Di dalam `SubprocVecEnv` setiap _worker_ memiliki prosesnya sendiri (dan kontributor sendiri), sehingga jendela _warm-up_ berlaku per-env — dengan `n_envs=4` dan `SPAWN_PROXIMITY_STEPS=50000`, rollout menerima total `4 × 50000 = 200k` transisi _warm-up_ (langkah absolut). Konsekuensi: atur `SPAWN_PROXIMITY_STEPS` berdasarkan **langkah per env**, bukan total langkah absolut, saat menggunakan paralelisme > 1.
- **Fallback:** bila bola Manhattan kekenyangan (ular sudah panjang, _radius_ tidak punya sel kosong) `_init_food()` jatuh ke `_random_empty_cell()` sehingga episode tetap dapat dimulai.
- **Filter jarak:** Manhattan (`|dr| + |dc| ≤ radius`), sehingga klaim "dapat dijangkau dalam ≤ _N_ langkah" berlaku tanpa tergantung rintangan.
- **Default `true`** di konfigurasi saat ini. Set ke `false` untuk menonaktifkan _warm-up_ dan kembali ke penempatan acak global sejak episode pertama.

### Contoh Nilai Default (Setelah Diubah)

```json
"SPAWN_PROXIMITY_ENABLED": true,
"SPAWN_PROXIMITY_RADIUS": 4,
"SPAWN_PROXIMITY_STEPS": 50000
```

Untuk mengaktifkan _warm-up_ dalam pelatihan berikutnya, set `SPAWN_PROXIMITY_ENABLED` ke `true` dan (opsional) sesuaikan radius / langkah. Rekomendasi awal: radius 4 sel + 50k env-steps cukup untuk mengajarkan asosiasi makanan tanpa membuat masalah navigasi jadi _trivially solvable_.

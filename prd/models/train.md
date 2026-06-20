# TRAINING INFRASTRUCTURE & TUI

Modul pelatihan menggunakan pendekatan paralelisasi (_SubprocVecEnv_) dan dioperasikan melalui antarmuka _Terminal User Interface_ (TUI) berbasis `curses` di skrip peluncur utama (`train.py`).

## Spesifikasi Antarmuka Curses (State Machine)

TUI beroperasi sepenuhnya via _keyboard_, menampilkan 5 menu utama:

1. **[Algorithm]:** PPO vs DQN (Kiri/Kanan).
2. **[Existing Model]:** Navigasi vertikal membaca isi direktori `saved_models/*.zip`. Menyertakan opsi `[0] <New Model>`.
3. **[Output Prefix]:** Input teks murni untuk prefix (Preview: `[prefix]_[algo]_level[X].zip`). Didukung fitur _Auto-Increment_ jika nama _file_ mengalami duplikasi.
4. **[Level Select]:** Level 1 hingga 5.
5. **[Parallelization]:** Kontrol utilitas CPU _core_ (1 hingga 8).

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

[Start Training] (Press Enter)

[Guide & Messages]

highlight focused option with inverse color. Use arrow keys to navigate and Enter to select.

for option if user press Enter: it will display the option list in the middle of the screen, user can navigate using arrow keys and select using Enter. The selected option will be highlighted.

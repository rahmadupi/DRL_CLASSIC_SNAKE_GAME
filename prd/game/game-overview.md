## GAME MECHANICS OVERVIEW

## Spesifikasi Arena
* **Dimensi:** Grid 2D berukuran 20x20 blok.
* **Win Condition (Terminal State):** Agen dinyatakan menang dan permainan dihentikan jika panjang tubuh ular mencapai luas maksimal peta (20x20 = 400).
* **Loss Condition:** Permainan berakhir jika kepala ular menabrak dinding (Kanal 1) atau menabrak tubuhnya sendiri (Kanal 2).

## Asimetri Kecepatan (Tick Skipping)
Untuk memastikan agen RL secara matematis mampu menangkap target yang melarikan diri (menghindari *infinite loop*), lingkungan menerapkan modifikasi *timestep*:
* **Kecepatan Ular:** Bergerak 1 blok pada setiap *step* komputasi.
* **Kecepatan Target Dinamis:** Bergerak 2 kali dalam setiap 3 *step* komputasi (jeda/berhenti pada iterasi ke-3).
* **Rasio Rasional:** Ular beroperasi 1.5x lebih cepat dari target dinamis.

## Antarmuka Visual
Visualisasi *gameplay* (`play_game.py`) dijalankan melalui Pygame dengan *scoring system* di sudut kiri atas layar. Opsi *timestep modifier* tersedia pada antarmuka GUI khusus untuk mode inspeksi (*Human/Watch AI*).

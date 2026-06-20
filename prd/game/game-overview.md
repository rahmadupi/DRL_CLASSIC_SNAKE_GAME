## GAME MECHANICS OVERVIEW

## Spesifikasi Arena

- **Dimensi:** Grid 2D berukuran 20x20 blok.
- **Win Condition (Terminal State):** Agen dinyatakan menang dan permainan dihentikan jika panjang tubuh ular mencapai luas maksimal peta (20x20 = 400).
- **Loss Condition:** Permainan berakhir jika kepala ular menabrak dinding (Kanal 1) atau menabrak tubuhnya sendiri (Kanal 2).

* **Inisialisasi:** Ular mulai di tengah peta dengan panjang awal 3
* **Penempatan Target:** Koordinat acak yang kosong, dengan logika penempatan khusus untuk target dinamis agar menghindari tumpang tindih dengan ular atau target lain.

## Asimetri Kecepatan (Tick Skipping)

Untuk memastikan agen RL secara matematis mampu menangkap target yang melarikan diri (menghindari _infinite loop_), lingkungan menerapkan modifikasi _timestep_:

- **Kecepatan Ular:** Bergerak 1 blok pada setiap _step_ komputasi.
- **Kecepatan Target Dinamis:** Bergerak 2 kali dalam setiap 3 _step_ komputasi (jeda/berhenti pada iterasi ke-3).
- **Rasio Rasional:** Ular beroperasi 1.5x lebih cepat dari target dinamis.

## Antarmuka Visual

Visualisasi _gameplay_ (`play_game.py`) dijalankan melalui Pygame dengan _scoring system_ di sudut kiri atas layar. Opsi _timestep modifier_ tersedia pada antarmuka GUI khusus untuk mode inspeksi (_Human/Watch AI_).

## UI FLOW

1. **Menu Utama:** Opsi untuk memulai permainan baru, Opsi: play as human ,watch AI play.
   Jika memilih "play as human", pengguna dapat mengendalikan ular menggunakan tombol panah keyboard. Jika memilih "watch AI play", pengguna dapat memilih model yang sudah dilatih untuk melihat performanya dalam lingkungan.
2. **Level Select:** Pilih tingkat kesulitan (Level 1-5) yang menentukan konfigurasi target (statis/dinamis).
3. **Time Delay Modifier:** Kecepatan Perubahan Step, modifier pace game.
4. **Gameplay Screen:** Input Tekan Tombol untuk Mulai flashing, Tampilan utama dengan skor, level, dan visualisasi grid.
5. **Game Over Screen:** Menampilkan hasil akhir (Menang/Kalah) dengan opsi untuk kembali ke Menu Utama.

## GAMEPLAY VISUAL

- **Ular:** Warna hijau dengan kepala yang lebih terang. Tubuhnya memiliki efek degradasi warna untuk menunjukkan urutan segmen.
- **Target Statis:** Warna merah cerah.
- **Target Dinamis:** Warna biru dengan efek jejak momentum (berubah warna secara dinamis berdasarkan posisi sebelumnya).
- **Dinding:** Warna abu-abu gelap. atau hitam. tipis tidak full block, hanya sebagai batas arena.
- **Skor:** Tampilkan di sudut kiri atas dengan font yang jelas, memperbarui secara real-time.

## UI Style
**Style** Neubrutal
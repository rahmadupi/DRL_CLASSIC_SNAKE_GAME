# TARGET ENTITIES BEHAVIOR

Target (makanan) dalam lingkungan dibagi menjadi dua varian dengan mekanika independen.

## 1. Target Statis
Entitas pasif yang dipijak di koordinat acak yang kosong. Terekam murni di Kanal 3.

## 2. Target Dinamis (Stochastic Momentum & Evasion)
Terekam di Kanal 4. Bergerak menggunakan *State Machine* probabilistik untuk mensimulasikan mangsa hidup dengan efisiensi komputasi maksimal $O(1)$.

* **State 1 (Momentum Jarak):** Target memilih jarak tempuh acak (3 hingga 8 blok) dan melaju lurus sejauh jarak tersebut tanpa henti.
* **State 2 (Rotasi & Evasion):** Setelah limit jarak tercapai, target memutuskan arah baru berdasarkan probabilitas terbobot:
  * *Standard Rotation:* Mengacak 4 arah (Atas, Bawah, Kiri, Kanan).
  * *Active Evasion (Probabilitas Rendah):* Fungsi *Greedy Evasion* dipanggil. Target mengevaluasi 4 kotak di sebelahnya secara instan dan memilih kotak dengan jarak *Euclidean* terjauh dari kepala ular (*no future pathfinding*).

## Proximity Collision Override
Jika target dinamis berjarak tepat 1 blok dari dinding atau target lain di jalur lurusnya (saat berada di State 1), ia wajib menghentikan siklus State 1 secara prematur dan langsung memicu State 2 pada *step* berikutnya untuk mencegah tumpang tindih data tensor.
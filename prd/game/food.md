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

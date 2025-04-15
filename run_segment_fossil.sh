conda activate sam2
echo"Lancement du Traitement pou 15485..."
python segment_fossil.py --bath_path "/home/killian/data2025/15485"

echo "Lancement du traitement pour 11478..."
python segment_fossil.py --base_path "/home/killian/data2025/11478"

echo "Lancement du traitement pour 13823..."
python segment_fossil.py --base_path "/home/killian/data2025/13823"

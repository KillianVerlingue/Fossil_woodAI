conda activate sam2

echo "Lancement du traitement pour TGV4..."
python segment_fossil.py --base_path "/home/killian/data2025/TGV4"

echo "Lancement du traitement pour TGV5..."
python segment_fossil.py --base_path "/home/killian/data2025/TGV5"

echo "Lancement du traitement pour 13823..."
python segment_fossil.py --base_path "/home/killian/data2025/13823"
~


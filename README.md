Mettre tous les csvs dans exps en suivant cette structure :
.
├── DAPP
│   ├── DAPP_CTRL&NMS_Session1_merged_Stats.xlsx
│   ├── DAPP_CTRL&NMS_Session2_merged_Stats.xlsx
│   └── DAPP_CTRL&NMS_Sessions12_merged_Stats.xlsx
├── FSS
│   └── FSS_test02.xlsx
├── HotAndColdPlate
│   ├── ColdPlate_CTRL&NMS_merged_Stats.xlsx
│   ├── HotColdRoom_CTRL&NMS_merged_AllPlates.xlsx
│   ├── HotPlate_CTRL&NMS_merged_Stats.xlsx
│   └── RoomTemp_CTRL&NMS_merged_stats.xlsx
└── Tickling
    ├── 2minutes Tickling
    │   ├── Tickling_2min_CTRL_Day10_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day1_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day2_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day3_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day4_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day5_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day6_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day7_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day8_merged_Stats.xlsx
    │   ├── Tickling_2min_CTRL_Day9_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day10_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day1_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day2_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day3_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day4_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day5_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day6_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day7_merged_Stats.xlsx
    │   ├── Tickling_2min_NMS_Day8_merged_Stats.xlsx
    │   └── Tickling_2min_NMS_Day9_merged_Stats.xlsx
    └── 30 sec before Tickling
        ├── Tickling_30sec_CTRL_Day10_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day1_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day2_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day3_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day4_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day5_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day6_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day7_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day8_merged_Stats.xlsx
        ├── Tickling_30sec_CTRL_Day9_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day10_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day1_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day2_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day3_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day4_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day5_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day6_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day7_merged_Stats.xlsx
        ├── Tickling_30sec_NMS_Day8_merged_Stats.xlsx
        └── Tickling_30sec_NMS_Day9_merged_Stats.xlsx

Commande à lancer:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 labelisation.py
python3 forced_swim.py
python3 plate.py
python3 tickling.py
python3 double_aversion.py
# Adversarial Adventures

- Author: Alan Leggitt
- Created: May 23rd, 2022

Repository for algorithmically generating images of fantasy characters 

## description

This project aims to generate novel images of fantasy characters using a conditional generative adversarial network (CGAN) architecture. Specific search terms are passed into google images via selenium and the resulting images are downloaded. A figure keypoint detection algorithm is applied. Detected figures are passed through a conditional variational auto encoder (CVAE). Weights from the CVAE are used to pretrain a CGAN (encoder -> discriminator; decoder -> generator), which is then trained on the input images. Novel images can be conditioned on search parameters as well as figure keypoints.  

### search parameters used for training

- SPECIES (dragonborn, drow, dwarf, elf, goblin, gnome, halfling, human, orc, tiefling)
- CLASS (barbarian, bard, cleric, druid, fighter, monk, paladin, ranger, rogue, sorcerer, warlock, wizard)
- GENDER (female, male, nonbinary)

## directory structure
```
README.md
requirements.txt
utils.py

notebooks/
- 01_Download_Dataset.ipynb
- 02_Inspect_Dataset.ipynb
- 03_Detect_Figures.ipynb
- 04_CVAE.ipynb
- 05_CGAN.ipynb

data/
- raw/
    - metadata.csv
    - SPECIES_CLASS_GENDER_RESULT.jpg
- figures/
    - metadata.csv
    - SPECIES_CLASS_GENDER_RESULT_FIGURE.pkl
outputs/
- models/
    - CVAE_TIMESTAMP.pth
    - CGAN_G_TIMESTAMP.pth
    - CGAN_D_TIMESTAMP.pth
- images/
    - CVAE_PERFORMANCE_TIMESTAMP.png
    - CVAE_SPECIES_CLASS_GENDER_RESULT_FIGURE.png
    - CGAN_PERFORMANCE_TIMESTAMP.png
    - CGAN_SPECIES_CLASS_GENDER_TIMESTAMP.png
```
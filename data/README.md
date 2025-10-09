# Data Directory

## Dataset Information

This project uses the NIKL Korean Hate Speech Dataset from the 2023 AI Competition.

## Dataset Structure

    NIKL_AU_2023_COMPETITION_v1.0/
    ├── train.csv         Training data
    ├── dev.csv           Validation data
    └── test.csv          Test data (unlabeled)

## Data Format

| Column | Description |
|--------|-------------|
| input  | Korean text |
| output | Label (0: Non-hate, 1: Hate) |

## Data Augmentation

Augmented data is stored in data/raw_aeda/

Method: AEDA - Randomly insert punctuation marks

## Privacy Note

Original dataset is not included in this repository.

## To Obtain Dataset

Contact NIKL (National Institute of Korean Language).

## Special Tokens

17 custom tokens for privacy masking:

&name&, &location&, &organization&, &account&, &address&, &number&, 
&site&, &email&, &phone&, &url&, &id&, &product&, &bank&, &card&, 
&date&, &money&, &company&

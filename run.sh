#!/usr/bin/env bash

# Create threshold, centroid and epsilon dictionaries
python -m src.scripts.create_meta_dict

# Train the triplet model
python -m src.scripts.train_triplet

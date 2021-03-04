#!/bin/bash
# This script will clean the folder of preprocessing data

preprocessing_data_hw_folder=/mnt/d/Utilisateurs/Alexandre/Repertoire_D/projet_super_resolution/data/marmoset_train_2/train_data
preprocessing_data_d_folder=/mnt/d/Utilisateurs/Alexandre/Repertoire_D/projet_super_resolution/data/marmoset_train_d_2/train_data


rm ${preprocessing_data_hw_folder}/ground_truth/*
rm ${preprocessing_data_hw_folder}/inputs/*

rm ${preprocessing_data_d_folder}/ground_truth/*
rm ${preprocessing_data_d_folder}/inputs/*

Remake of SegSRGAN code : https://github.com/koopa31/SegSRGAN

Code from this paper :
Super-resolution and segmentation using generative adversarial networks — Application to neonatal brain MRI paper

Authors :  
Quentin Delannoy 1  
Chi-Hieu Pham 2  
Clément Cazorla 1   
Carlos Tor-Díez 2  
Guillaume Dollé 3  
Hélène Meunier 4  
Nathalie Bednarek 1, 4  
Ronan Fablet 5  
Nicolas Passat 1  
François Rousseau 2 

1 CRESTIC - Centre de Recherche en Sciences et Technologies de l'Information et de la Communication - EA 3804  
2 LaTIM - Laboratoire de Traitement de l'Information Medicale  
3 LMR - Laboratoire de Mathématiques de Reims  
4 Service de médecine néonatale et réanimation pédiatrique, CHU de Reims  
5 Lab-STICC - Laboratoire des sciences et techniques de l'information, de la communication et de la connaissance  


## Utilisation :  

run_model.py :  
> python run_model.py -n run_model -f D:\Projets\srm4bmri\dataset\1010\hr1010.nii.gz -o D:\Projets\srm4bmri\outputs\results -m D:\\Projets\\srm4bmri\\training_folder\\checkpoints\\training_10_epochs\\ -ps 64 64 64

On Osirim :  
Modify the file build_env.sh to have the wanted path, and wanted files names.
Launch it to create the environment automatically :  
> sbatch build_env.sh  

For each dataset, modify the file build_dataset.sh with the wanted parameters.
Launch it to create a dataset given a csv file and a set of data :  
> sbatch build_dataset.sh

Finnaly, launch a train with (specifying the dataset, the number of epochs etc. in the file) : 
(mri_test is optionnal and it is the mri used to have a visualisation between each epoch)
> sbatch train.sh


Launch a model with (specifying the model in the file) :  
> sbatch run_model.sh

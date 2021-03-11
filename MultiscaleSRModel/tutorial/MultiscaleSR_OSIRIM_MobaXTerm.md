# Use Multiscale Super-Resolution Model on OSIRIM with MobaXTerm

## Important to know

In OSIRIM you have two repositories, your home repository 
(`/users/<projectname>/<username>`) and your project repository 
(`/projects/<projectname>`). Make sure to save the scripts and environment in 
your home repository and your data and outputs in your project repository.

Moreover the main script is `srrecnn.py`. It is this one you may pass to the
`script` variable in the three job files presented here: `prepare.sh` 
`launch_training.sh` and `launch_testing.sh`. 

## Get the scripts

1.  Download the folder `MultiscaleSRModel` from 
    [here](https://github.com/adlane98/SRM4BMRI/tree/main/MultiscaleSRModel).
2.  Put this folder on your session in OSIRIM wherever you want 
    (with MobaXTerm you can easily drag & drop).

## Preparation of the environment

3.  You need to create a virtual environment on your OSIRIM session:
    1. First of all you need to open a shell in a container. Use the 
       `CUDA11_tf2-NGC-20-06-py3.sif` container:
       
        `singularity shell /logiciels/containerCollections/CUDA11/tf2-NGC-20-06-py3.sif`

    2. Then you can create your virtual environment in your home repository
       with the option `--system-site-packages` which makes you able to use
       all the pre-installed packages in the container.
    
        `virtualenv --system-site-packages ~/path/to/ENVNAME`

    3. To avoid an error telling there is no more space left, execute those 
       commands:
    
    ````shell
    cd
    mkdir localTMP
    TMPDIR=~/localTMP
    TMP=$TMPDIR
    TEMP=$TMPDIR
    export TMPDIR TMP TEMP
    ````
    
    4. Then you can activate your environment and install the requirements:
    
    ````shell
    source /path/to/ENVNAME/bin/activate
    pip install -r /path/to/MultiscaleSRModel/requirements.txt
    exit #To exit the container
    ````
    
## Creation of patches

4.  With MobaXTerm editor open the file 
    `/path/to/MultiscaleSRModel/pathes.json`
    (right-click on the file -> `Open with default text editor`).
    1.  Indicate the value of the key `metadata`: it is the folder where you
        want to store the json file summarising parameters you will choose for 
        generating patches. 
        This folder needs to be under `/projets/<projectname>/`.
        
    2.  Indicate the value of the key `hdf5`: it is the folder where you
        want to store hdf5 files containing the patches. 
        This folder needs to be under `/projets/<projectname>/`.
    
5.  With MobaXTerm editor open the file 
    `/path/to/MultiscaleSRModel/scripts/prepare.sh`. 
    This file is a SLURM job that you will run in a cluster. With this tutorial, 
    there is a PowerPoint describing the structure of this kind of file.
    **You need to specify some parameters for your session.**
    Check this file for modifying those parameters like the cluster, your email,
    output and error flows, etc...
    
6. In this file go to the last line, and after the option `--prepare`, 
   define your own values:
   
| Option name | Description | Mandatory / Default value |
| :--------------- |:---------------| :-----:|
| `--prepare`  | Indicates that we want to generate patches | No value - Mandatory |
| `--mri`  | Folder where are stored all volumes we want to slice | Mandatory |
| `--sigma` | Value of the noise's standard deviation with which we want to contaminate patches. `-1` if we want a random value for each patch. | 1.0 |
| `--scale`, `-s` | Scale factor we want to apply on volumes. Append mode: `-s 2,2,2 -s 3,3,3`. | 2,2,2 |
| `--order` | Order of the spline interpolation on the downsampling phase | 3 |
| `-p`, `--patchsize` | Patch size. Same size on x, y and z axes. | 21 |
| `--stride` | Patch stride of the extraction step. | 10 |
| `--samples` | Maximum of patches per volume. | 3200 |

7.  When all is good you can run the job and execute the following command 
    `sbatch /path/to/MultiscaleSRModel/prepare.sh`.
    
8.  With the command `squeue` you can see if your model is correctly running.

9.  If there is an error, you can check the file you specified as the error
    flow in the SLURM job.

10.  Patches are stored in the folder you specified in step `4.ii`. 
   
11.  In this folder, there is a `.txt` file named with the hour you executed the
    job. It contains the paths to all the patches grouped by volume. 
    
12.  Like said in step `4.i`, you can find the summary of parameters you have 
     chosen, in the folder you indicated in `metadata` key: 
     `YYYYMMDD-hhmmss_preproc_parameters.json`.
   
## Training

13. Now that patches are generated you can use them for training the model.
    Open `/path/to/MultiscaleSRModel/launch_training.sh`, and after the option 
    `--train` specify model options:
    
| Option name | Description | Mandatory / Default value |
| :--------------- |:---------------| :-----:|
| `--train`  | Indicates that we want to train the model | No value - Mandatory |
| `-i`, `--input`  | Path of `.txt` file described in step `11`. | The last one in the path of `hdf5` key |
| `-l`, `--layers` | Number of convolutional layers | 10 |
| `--numkernels` | Number of kernel per layer | 64 |
| `-k`, `--kernel` | Kernel size. | 3 |
| `--epochs` | Number of epochs. | 20 |
| `-b`, `--batch` | Batch size. | 64 |
| `--adam` | Adam optimizer learning rate. | 0.0001 |

14.  You can run the following command 
    `sbatch /path/to/MultiscaleSRModel/launch_training.sh`.

15. In `metadata` folder you will find loss and psnr curves, weights backup, 
    and a json file summarising parameters you used for training the model:
    `YYYYMMDD-hhmmss_training_parameters.json` with the hour at which you
    executed your model.
    
16. In this folder you will also find a backup of the model trained in the form
    of a folder: `YYYYMMDD-hhmmss_model`.
    
## Testing

17. Open `/path/to/MultiscaleSRModel/launch_testing.sh`, and after the option 
    `--test` specify testing options: 
    
| Option name | Description | Mandatory / Default value |
| :--------------- |:---------------| :-----:|
| `--test`  | Indicates that we want to test the model. | No value - Mandatory |
| `--model`  | Path of the model described in step `15`. | Mandatory |
| `--testinput`  | Folder of the images we want to test. | Mandatory |
| `--output` | Folder where to stored output files. | Mandatory |

18.  You can run the following command 
    `sbatch /path/to/MultiscaleSRModel/launch_testing.sh`.

19. You will find the results in the folder you indicated at `--ouput` option.
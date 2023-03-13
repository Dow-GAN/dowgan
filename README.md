# Project DowGAN

# Exploring Generative Adversarial Networks for Data Augmentation

This project uses generative adversarial networks for data generation for time-series data.

## Description

Collecting large amounts and high resolution of data in industrial fields is often costly and time intensive. General adversarial networks (GANs) are known for augmenting images. We are exploring the potential for GANs to augment time-series data. 

This project is a collaboration between the University of Washington and Dow Chemical.

## Getting Started

### Installing/Modifications

* Clone github repo into home directory
* Create environment by executing at the command line:
```
conda env create -n dowgan-env --file environment.yml
```
* To activate environment: `conda activate dowgan-env`
* To deactivate environment: `conda deactivate dowgan-env`
* To see the list of all conda environments: `conda info --envs`

### Dependencies

* To create the environment for this github repo

### Executing program

* Clone github repo into home directory
* Navigate to '../dowgan/dowgan' folder
* To run GAN on default Hungary Chicken Pox Data, execute at the command line:
```
python3 dataloader.py
```
* Modify parameters and preferences for GAN model by editing `dataloader.py` file parameters 


## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

Daniel Kuo - [@Whast1225](https://github.com/Whast1225)  
Emily Miura-Stempel - [@emimiura](https://github.com/emimiura)  
Emily Nishiwaki - [@emynish](https://github.com/emynish)  
Arty Timchenko - [@atimch](https://github.com/atimch)

## Version History

* 1.0
    * Initial Release - Winter Quarter 2023 CHEM E 545/546

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [DomPizzie](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)


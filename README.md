# Project DowGAN

# Exploring Generative Adversarial Networks for Data Augmentation

This project uses generative adversarial networks for data generation for time-series data. A Gantt chart of the project can be found here: [Dow GAN Gantt Chart](https://docs.google.com/spreadsheets/d/1TDkpIP71Qa-njRVNn1pCFSOW9tKn3kIfl4yciAdwVgk/edit?usp=sharing)

## Description

Collecting large amounts and high resolution of data in industrial fields is often costly and time intensive. General adversarial networks (GANs) are known for augmenting images. We are exploring the potential for GANs to augment time-series data. 

This currently only works on the `hungary_chickenpox.csv` data

This project is a collaboration between the University of Washington and Dow Chemical.

## Getting Started

### Installing & Dependencies

* Clone github repo into home directory
* Navigate into the dowgan repo from your Home directory using `cd dowgan`
* Create environment by executing at the command line for mac or windows/pc respectively:
```
conda env create -f environment_mac.yml
```
```
conda env create -f environment_windows.yml
```
* To activate environment: `conda activate dowgan`
* To deactivate environment: `conda deactivate dowgan`
* To see the list of all conda environments: `conda info --envs`
* To update the environment: 
    * Make any changes required to environment.yml file
    * Then run: `conda env update --file environment_mac/windows.yml --prune`
    * Then make sure to `conda activate dowgan` again to successfully update environment.

### Executing program

* First follow the steps above in Installing & Dependencies
* Once the dowgan environment is activated, navigate into the scripts folder `cd scripts`
* To run GAN script called 'dataloder.py' on the default 'Hungary Chicken Pox' Data, execute at the command line:
```
python3 dataloader.py
```
* Modify parameters and preferences for GAN model by editing `dataloader.py` script file parameters 


## Help

Contact authors listed below for help.

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


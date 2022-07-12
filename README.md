# Schau mir in die Augen

People recognition based on eye movements.

Code for the paper: 
_**Robustness of Eye Movement Biometrics Against Varying Stimuli and Varying Trajectory Length**. Christoph Schröder, Sahar Mahdie Klim Al Zaidawi, M.Sc. Martin H.U. Prinzler, Sebastian Maneth, Prof. Dr. Gabriel Zachmann, CHI 2020, Honolulu, Hawaiʻi, April 25 - 30, 2020._

`@inproceedings{Schroeder-2020-RobustnessEye,
  author = {Schröder, Christoph and Al Zaidawi, Sahar Mahdie Klim and Prinzler, Martin H.U. and Maneth, Sebastian and Zachmann, Gabriel},
  title = {Robustness of Eye Movement Biometrics Against Varying Stimuli and Varying Trajectory Length},
  year = {2020},
  isbn = {9781450367080},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3313831.3376534},
  doi = {10.1145/3313831.3376534},
  booktitle = {Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems},
  pages = {1–7},
  numpages = {7},
  keywords = {eye tracking, gaze detection, eye movement biometrics},
  location = {Honolulu, HI, USA},
  series = {CHI ’20}
}`


## Setup development environment

1. Make sure to use Python 3
2. Install the requirements from `requirements.txt`
3. Download the datasets (see Datasets section)
4. Run make test for some basic sanity checks
5. Train and evaluate a classifier:
- `cd scripts && $(PYTHON) train.py --method score-level --dataset bio-tex --clf rbfn`
- `cd scripts && $(PYTHON) evaluation.py --method score-level --dataset bio-tex --clf rbfn` 

## Datasets
Extract the datasets into the following directories. The folder structure is documented data/files.txt which was generated with the `tree data/` command so you can check if the structure matches.
### BioEye
BioEye 2015 - Competition on Biometrics via Eye Movements

dir: `data/BioEye2015_DevSets`

### MIT
Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba. Learning to Predict where Humans Look [ICCV 2009]

Available here:
http://saliency.mit.edu/datasets.html

dir: `data/where_humans_look`


## License
This original work is copyright by University of Bremen.

Any software of this work is covered by the European Union Public Licence v1.2. To view a copy of this license, visit eur-lex.europa.eu.
Any other assets (3D models, movies, documents, etc.) are covered by the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit creativecommons.org. If you use any of the assets or software to produce a publication, then you must give credit and put a reference in your publication.
If you would like to use our software in proprietary software, you can obtain an exception from the above license (aka. dual licensing). Please contact zach@cs.uni-bremen dot de.

## Cite
_**Robustness of Eye Movement Biometrics Against Varying Stimuli and Varying Trajectory Length**. Christoph Schröder, Sahar Mahdie Klim Al Zaidawi, M.Sc. Martin H.U. Prinzler, Sebastian Maneth, Prof. Dr. Gabriel Zachmann, CHI 2020, Honolulu, Hawaiʻi, April 25 - 30, 2020._

`@inproceedings{Schroeder-2020-RobustnessEye,
  author = {Schröder, Christoph and Al Zaidawi, Sahar Mahdie Klim and Prinzler, Martin H.U. and Maneth, Sebastian and Zachmann, Gabriel},
  title = {Robustness of Eye Movement Biometrics Against Varying Stimuli and Varying Trajectory Length},
  year = {2020},
  isbn = {9781450367080},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3313831.3376534},
  doi = {10.1145/3313831.3376534},
  booktitle = {Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems},
  pages = {1–7},
  numpages = {7},
  keywords = {eye tracking, gaze detection, eye movement biometrics},
  location = {Honolulu, HI, USA},
  series = {CHI ’20}
}`

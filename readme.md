<center>
    <h2>
        Computer Vision Project
    </h2>
</center>

---

<center>
    <h3>
        Object detection on LIDAR images from an autonomous car
    </h3>
</center>

---


#### by Zaim Imran, Martin Johannes Nilsen and Max Torre Schau

## Introduction
This is the final project in the subject TDT 4265 - Computer Vision and Deep Learning. 


## Tutorials from handout code
- [Introduction to code](notebooks/code_introduction.ipynb).
- [Dataset setup](tutorials/dataset_setup.md) (Not required for TDT4265 computers).
- [Running tensorboard to visualize graphs](tutorials/tensorboard.md).


## Setup

After having successfully cloned the repository, you will need to make sure that you have the required python modules installed.

### Virtual environment
We recommend you using a virtual environment, which can be created with
```
virtualenv -p /usr/bin/python3 venv
```
You can now place all the required modules in this virtual environment instead of globally on your computer. 

In your terminal you can access this using
```
source venv/bin/activate
```
*Note that this requires the package virtualenv on your computer*
<details>
    <summary>HowTo</summary>
    
    python -m pip install --user virtualenv

</details>

### Install required packages
Then, install the required packages with
```
pip install -r requirements.txt
```

## Dataset installation
The dataset can be downloaded using the script
```
python scripts/update_tdt4265_dataset.py
```

## Dataset train
For training the data, you can simply define the configurations in a config-file, and run

```
python train.py configs/task2_1.py
```

## Torchinfo
The project utilizes torchinfo for printing out the model and parameters. Simply run 
```
python train.py configs/task2_1.py --torchinfo-only
```
To print out this for the given model.


## Dataset exploration 
We have provided some notebooks for the parts covering dataset exploration. See the folder called `dataset_exploration`.

## Dataset visualization

We have also created a script visualizing images with annotations. To run the script, do 

```
python -m dataset_exploration.save_images_with_annotations
```

By default, the script will print the 500 first train images in the dataset, but it is possible to change this by changing the parameters in the `main` function in the script.

## Qualitative performance assessment

To check how the model is performing on real images, check out the `performance assessment` folder. Run the test script by doing:

```
python -m performance_assessment.save_comparison_images <config_file>
```

If you for example want to use the config file `configs/tdt4265.py`, the command becomes:

```
python -m performance_assessment.save_comparison_images configs/tdt4265.py
```

This script comes with several extra flags. If you for example want to check the output on the 500 first train images, you can run:

```
python -m performance_assessment.save_comparison_images configs/task2_4.py --train -n 500
```

### Test on video:
You can run your code on video with the following script:
```
python -m performance_assessment.demo_video configs/tdt4265.py input_path output_path
```
Example:
```
python3 -m performance_assessment.demo_video configs/tdt4265.py Video00010_combined.avi output.avi
```
You can download the validation videos from [OneDrive](https://studntnu-my.sharepoint.com/:f:/g/personal/haakohu_ntnu_no/EhTbLF7OIrZHuUAc2FWAxYoBpFJxfuMoLVxyo519fcSTlw?e=ujXUU7).
These are the videos that are used in the current TDT4265 validation dataset.



## Bencharking the data loader
The file `benchmark_data_loading.py` will automatically load your training dataset and benchmark how fast it is.
At the end, it will print out the number of images per second.

```
python benchmark_data_loading.py configs/tdt4265.py
```

## Runtime analysis
The file `runtime_analysis.py` will automatically will test the inference speed and FPS for the given model.

```
python runtime_analysis.py configs/task2_1.py
```


## Uploading results to the leaderboard:
Run the file:
```
python save_validation_results.py configs/tdt4265.py results.json
```
Remember to change the configuration file to the correct config.
The script will save a .json file to the second argument (results.json in this case), which you can upload to the leaderboard server.

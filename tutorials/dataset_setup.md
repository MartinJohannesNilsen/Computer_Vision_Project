# Setting Up Dataset
Note that all datasets are downloaded on the TDT4265 compute server, this is for downloading on own computer.

## TDT4265 Dataset
To take a peek at the dataset, take a look at [visualize_dataset.ipynb](../notebooks/visualize_dataset.ipynb).

#### Getting started
**Download/Setup:**
```
python3 scripts/update_tdt4265_dataset.py
```
This will automatically download the dataset on local computers and create symlinks to the folder data if you are on TDT4265 compute servers.

#### Dataset Information
Label format follows the standard COCO format (see [here](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) for more info).
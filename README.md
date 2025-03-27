## Data Download

To download datasets, run the following command. Replace Facebook and Wiki with your desired dataset names.

    >>> python data_processing/data_download.py --datasets Facebook Wiki

To split the dataset into training and test sets, specify the dataset and the desired split ratio (e.g., 30% train data):

    >>> python data_processing/train_test_split.py --dataset Facebook --ratio 0.3 
# TablaGharanaRecog
Implementation of "TABLA GHARHANA RECOGNITION FROM AUDIO MUSIC RECORDINGS OF TABLA SOLO PERFORMANCES", ISMIR 2021

1. Data Preparation 
2. 1.data_segment_and_store_with_label.py : Take the Data directories and search for the wavefile. After that segment the wavfile for specified duration. At the end it will store in the h5 format file.
  help: python data_segment_and_store_with_label.py 
3. train.py  :  Main train module. Takes the data(.h5) files and starts training. Best checkpoint will be stored in models directory

4. model.py and dataLoader.py sub modules of train.py

5. test_prob.py : performs inference on development and test data and print the confusion matrix and report for only test data and store the "true predicted probabilities"

6. segment_test_h5.py : Split the 10 sec raw samples to 5sec and 3 sec raw samples 


# Dataset
Dataset information will be shared on request. The authors do not have the right to share a few of the audio recordings as they are from online sources. The details related to the link of the audio source and the time stamps are given. The dataset details can be accessed from https://sites.google.com/view/gowriprasad-phd-thesis-iitm/datasets/gharana-dataset


# Reference 
Please cite the following paper when this code is used for research
R Gowriprasad, V Venkatesh, Hema A Murthy, R Aravind, K S R Murty, "Tabla Gharana Recognition from Audio Recordings of Tabla Solo Performances," Proc. International Society for Music Information Retrieval Conference ISMIR 2021, Nov. 2021. DOI: 10.5281/zenodo.5624630

R Gowriprasad, R Aravind and Hema A Murthy, “Structural Segmentation and Labeling of Tabla Solo Performances,” Journal of New Music Research, 2023. https://doi.org/10.1080/09298215.2023.2265912


To get the information about the dataset, contact gowriprasadmysore@gmail.com/ee19d702@smail.iitm.ac.in

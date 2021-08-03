# TablaGharanaRecog
Implementation of "TABLA GHARHANA RECOGNITION FROM AUDIO MUSIC RECORDINGSOF TABLA SOLO PERFORMANCES"

1. Data Preparation 
2. 1.data_segment_and_store_with_label.py : Take the Data directories and search for the wavefile. After that segment the wavfile for specified duration. At the end it will store in the h5 format file.
  help: python data_segment_and_store_with_label.py 
3. train.py  :  Main train module. Takes the data(.h5) files and starts training. Best checkpoint will be stored in models directory

4. model.py and dataLoader.py sub modules of train.py

5. test_prob.py : performs inference on development and test data and print the confusion matrix and report for only test data and store the "true predicted probabilities"

6. segment_test_h5.py : Split the 10 sec raw samples to 5sec and 3 sec raw samples 

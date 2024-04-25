How to Run
=========================
    -load train.csv and test.csv to this folder, both of them should stay in the same folder containing this readme

    -to any of the 4 files
        -python <filename.py> (no arguments neccessary)
        -in place of  <filename.py>
            – svd.py: Train the word embeddings using SVD method and savethe word vectors.
            – skip-gram.py: Train the word embeddings using Skip gram method(with negative sampling) and save the word vectors.
            – svd-classification.py: Train any RNN on the classification task using the SVD word vectors.
            – skip-gram-classification.py: Train any RNN on the classification task using the Skip-Gram word vectors.

    -keep the folder and file structure as provided, the code will run smoothly

    

Assumptions
================
    -train.csv and test.csv is not included in folder, it must be provided
    -Since embedding sizes were large, they are not included, but uploaded to drive, link provided below:
        -https://drive.google.com/file/d/1lZX6-Njpgj8dM9cZR3fa9obfXOWeeCnT/view?usp=drive_link

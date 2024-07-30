Python requirements:

    keras
    keras_cv
    tensorflow
    tensorboard
    numpy
    matplotlib
    pillow
    datetime
    scikit-learn
You may also have to install tensorboard seperately on linux.

Windows tensorflow no longer supports using the gpu, so to speed things up, you may have to downgrade tensorflow and keras (this is problematic for many reasons) or use linux (WSL is perfectly fine) 

Alternatively, you can just use the CPU and skip installing cuda, cudnn, and everything else tensorflow needs which is sometimes a large pain

Description of folder structure:

    /classifier/
        /training/  < PUT TRAINING DATA IN THIS FOLDER
            /[category1] these categories can be named anything and have as many as you want
            /[category2]  
            /[category[n]]
        /input/     < PUT DATA THAT YOU WANT TO SORT HERE 
            /[folder1] these can also be named anything
            /[folder2] the code that turns the training data into a dataset is the same as what turns the
            /[folder[n]] data that is desired to be sorted so it must be in a subfolder (the names of the folder doesnt change much)
        /output/   > DATA GETS SORTED INTO CATEGORIES FROM /training/ (dont put files here)
            /[category1]
            /[category2]  
            /[category[n]]
        /logs/
            /fit/ this is for tensorboard, please start tensorboard by running the shell command 
                                            "tensorboard --logdir=logs/fit" 
                                            in the root classifier directory (through the shell, not through python)

for any questions, comments, or concerns, the easiest way to reach me is on discord @toothless890

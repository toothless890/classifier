    Description of folder structure
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

# DCCC-LSTM
cs570_2017_spring_term project

## Required
[mnist.h5](https://drive.google.com/open?id=0B3kZyL62Zw6vOUJVUE94R3FjVjQ) (need to be installed & located at "data/")

## Folders
DC-LSTM/	Deep ConvLSTM

DCCC-LSTM/	Deep Conditioned Composite ConvLSTM 

DCComp-LSTM/	Deep Composite ConvLSTM 

DCCond-LSTM/	Deep Conditioned ConvLSTM 

GAN/	GAN

basicLSTM/ simple single layer LSTM

## Weight save & load

### How to save
- train_example: will be saved into weights folder
- train_example_conv: will be save into weights_conv folder

### How to restore weights
argv[1] gets a weight file path, without a file extension.

loss~.~~~~~~

```bash
$ python train_example.py weights/2017-05-10_22:51:26.218343__step0__loss2839.130859
$ python train_example_conv.py weights_conv/2017-05-10_22:51:26.218343__step0__loss2839.130859
```

## Video Evaluation

```bash
$ python evaluation.py [weight-set]
$ python evaluation_conv.py [weight-set]
```

 Video folder will be created on execution. Videos will be saved into the folder.
 
 Test data set :  https://drive.google.com/open?id=0B4DyTsz0E9HFakRhb2FPVzJieUE
 We edited the original moving mnist dataset so that the sequency of array is different for our convenience.
 

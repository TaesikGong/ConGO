# ConGO
cs570_2017_spring_term project

## Required
[mnist.h5](https://drive.google.com/open?id=0B3kZyL62Zw6vOUJVUE94R3FjVjQ) (need to be installed & located at "data/")

## Branches
- master branch: (1)train_example: simple LSTM, (2) train_example_conv: Conv+deconv + convLSTM
- condDec branch: master/train_example_conv + conditioned decoder
- composite branch : composite decoder
- ccc branch : composite + conditioned decoder
- GAN branch: GAN

## Weight save & load

### save하기
- train_example: weights 폴더에 저장됨
- train_example_conv: weights_conv 폴더에 저장됨

### 저장된 weight restore하기
argv[1]에 weight 파일명을 입력하는데, 확장자명을 빼야 함.

loss~.~~~~~~ (소수점 밑 6자리까지)

```bash
$ python train_example.py weights/2017-05-10_22:51:26.218343__step0__loss2839.130859
$ python train_example_conv.py weights_conv/2017-05-10_22:51:26.218343__step0__loss2839.130859
```

## Video Evaluation

```bash
$ python evaluation.py [weight-set]
$ python evaluation_conv.py [weight-set]
```

 기존 프로그램과 실행방법 동일하며, 실행시 Video폴더가 생성되며, 그안에 비디오들이 저장됨.
 
 Test data set :  https://drive.google.com/open?id=0B4DyTsz0E9HFakRhb2FPVzJieUE
 
 (편의상 배열 형태만 수정해서 저장한 것으로, 기존 Moving mnist 데이터와 배열이 다름)

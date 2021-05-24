# LRP_MITBIH_Arrhythmia

XAI 알고리즘 중 하나인 LRP 알고리즘을 이용하여 Arrhythmia detection을 한 내용이다.

LRP tool (https://github.com/sebastian-lapuschkin/lrp_toolbox), Arrhythmia dataset(https://www.kaggle.com/shayanfazeli/heartbeat)

LRP tool에서는 example로 이미지를 사용하였기 때문에 CNN의 경우 filter size = 4 * 1 등으로 사용하였다.

```python
maxnet = modules.Sequential([
        modules.Convolution(filtersize=(4, 1, 1, 2), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 2, 4), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 4, 8), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 8, 16), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(4, 1, 16, 8), stride=(1, 1)), \
        modules.MaxPool(pool=(2, 1), stride=(2, 1)), \
        modules.Convolution(filtersize=(2, 1, 8, len(label)), stride=(1, 1)), \
        modules.Flatten(), \
        modules.SoftMax()
    ])
```

config.py 파일에 CNN 모델과 NN 모델을 수정할 수 있다.

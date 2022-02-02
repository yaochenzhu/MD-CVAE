# MD-CVAE: Mutually-Regularized Dual Collaborative Variational Auto-encoder for Recommendation Systems

 The codes are associated with the following paper:
 >**Mutually-Regularized Dual Collaborative Variational Auto-encoder for Recommendation Systems,**  
 >Yaochen Zhu and Zhenzhong Chen,  
 >*The Web Conference (WWW) 2022.*


## Environment

 The codes are written in Python 3.6.5.  

- numpy == 1.16.3
- pandas == 0.21.0
- tensorflow-gpu == 1.15.0
- tensorflow-probability == 0.8.0

## Datasets

 The raw features of the established movielen-sub dataset can be obtained here: [[Google Drive]](https://drive.google.com/file/d/1GOLFs494n2lW8PGPVzZ8N1_ky_8bZpwD/view?usp=sharing), [[Baidu]](https://pan.baidu.com/s/1iecjErD59EidAl9yO3VRbw?pwd=r8f0) (PIN:r8f0)

 The processed datasets can be found here: [[Google Drive]](https://drive.google.com/file/d/1P7QmMG3R3Jk_PpW53NGiu26qulvnyVrw/view?usp=sharing), [[Baidu]](https://pan.baidu.com/s/1sACXBamQnGGT6MBlH-C6rQ?pwd=l8wr) (PIN:l8wr)

 For usage, please unzip the processed datasets and copy them into the data folder.

## Examples to run the codes
### 1. To reproduce the in-matrix prediction results:
- **Pretrain the dual item content embedding VAE**: 

    ```python pretrain_vae.py --dataset Name --split [0-9]```
- **Iteratively train the collaborative and content VAEs**:

    ```python train_vae.py --dataset Name --split [0-9]```
- **Evaluate the model and summarize the results into a pivot table**
    
    ```python predict.py --dataset Name --split [0-9]```
    
    ```python summarize.py```

### 2. To reproduce the out-of-matrix prediction results:
- First, please change to the cold_start folder.

- Download the processed data [[Google Drive]](https://drive.google.com/file/d/1mHJUgkYWed2v8DRfA3NFITtgCFfRt1LT/view?usp=sharing), [[Baidu]](https://pan.baidu.com/s/1YdJEkxQMNgaw-cQsj3w0FA?pwd=f4f7) (PIN:f4f7)

- The way to run the codes and summarize the results is similar to the in-matrix case.

**For advanced argument usage, run the code with --help argument.**

## Reference

### if you find the codes helpful, please kindly cite our paper. Thanks!

    @inproceedings{MDCVAE-WWW2022,
      title={Mutually-Regularized Dual Collaborative Variational Auto-encoder for Recommendation Systems},
      author={Zhu, Yaochen and Chen, Zhenzhong},
      booktitle={Proceedings of the ACM Web Conference 2022},
      year={2022},
    }    

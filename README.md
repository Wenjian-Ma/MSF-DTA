# MSF-DTA
Code for paper "Predicting drug-target affinity by learning protein knowledge from biological networks"
---

Dependencies
---

python == 3.7.11

pytorch == 1.7.1

PyG (torch-geometric) == 2.0.2

rdkit == 2020.09.5

numpy == 1.21.2

Data preparation (password: 1234)
---
1. 
   Unzipping all ''**.rar files**'' to their path.
2. 
   1) PPI network are available at [Link](https://pan.baidu.com/s/1M8UTTEzJ6cvv322cCD4vJQ "password:1234").
   2) Unzipping ''**ppi.rar**'' to ''**MSF-DTA/data/networks/**''.
   3) Trained latent representations of 18,552 proteins are available at [Link](https://pan.baidu.com/s/1RmwYTlhQFrwl6zTVEUZiLg "password:1234").
   4) Unzipping ''**embeddings.rar**'' to ''**MSF-DTA/data/**''.
3. data splitting for Davis and KIBA.

   ```python create_data.py```
4. data splitting for Human

   ```python create_data_for_CPI.py```

DTA task (Davis and KIBA).
---
1) Evaluating trained model by us on Davis dataset.

    ```python test.py 0 0 0```

2) Evaluating trained model by us on KIBA dataset.

    ```python test.py 1 0 0```

3) If you want to train your own model on Davis dataset.

    ```python training_validation.py.py 0 0 0```

4) If you want to train your own model on KIBA dataset.

    ```python training_validation.py.py 1 0 0```
  
CPI task (Human).
---
Training and testing are combined.

```python train_for_CPI.py```

latent representations.
---
if you want to train your own latent representations.

```python embeddings_gen.py```

Cite our work
---
if you use the conclusion, code, or data in our work, please cite:
```
@ARTICLE{10027191,
  author={Ma, Wenjian and Zhang, Shugang and Li, Zhen and Jiang, Mingjian and Wang, Shuang and Guo, Nianfan and Li, Yuanfei and Bi, Xiangpeng and Jiang, Huasen and Wei, Zhiqiang},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Predicting Drug-Target Affinity by Learning Protein Knowledge From Biological Networks}, 
  year={2023},
  volume={27},
  number={4},
  pages={2128-2137},
  keywords={Proteins;Protein engineering;Drugs;Amino acids;Feature extraction;Predictive models;Knowledge engineering;Drug-target affinity;variational graph auto-encoders;graph convolutional network;protein- 
  protein interaction},
  doi={10.1109/JBHI.2023.3240305}}
```

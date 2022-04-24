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

Data preparation
---
1. 
   Unzip all ''**.rar files**'' to their path.
2. 
   1) PPI network are available at [Link](https://pan.baidu.com/s/1M8UTTEzJ6cvv322cCD4vJQ "password:1234").
   2) Unzip ''**ppi.rar**'' to ''**MSF-DTA/data/networks/**''.
   3) Trained latent representation of 18,552 proteins are available at [Link](https://pan.baidu.com/s/1RmwYTlhQFrwl6zTVEUZiLg "password:1234").
   4) Unzip ''**embeddings.rar**'' to ''**MSF-DTA/data/**''.
3. data split for Davis and KIBA.

   ```python create_data.py```
4. data split for Human

   ```python create_data_for_CPI.py```


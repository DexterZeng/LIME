# LIME
The source code for the paper: On entity alignment at scale.

Credit to the [RREA repo](https://github.com/MaoXinn/RREA). 

## Dependencies

* Python=3.6
* Tensorflow-gpu=1.13.1
* Scipy
* Numpy
* Scikit-learn
* python-Levenshtein

## Datasets
The original datasets are obtained from [DBP15K dataset](https://github.com/nju-websoft/BootEA),  [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_1_trans_goo: entities in source KG (ZH) with translated names;
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links for testing/validation;
* sup_ent_ids: entity links for training;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;


## Running
For evaluation on the small and medium-sized datasets without the partition strategies: 
* Unpackage data.zip
* Configure the settings in run.sh.
* Then run

```
bash run.sh 
```

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit  when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to zengweixin13@nudt.edu.cn.

## Citation

To be added.

## Evaluation on the large-scale datasets
For evaluation on the large-scale dataset, open the Large-scale directory. 
Unpackage the DBP15K.zip, DWY100K.zip.
The size of the FB_DBP_2M dataset is too large. You can download it [here](https://1drv.ms/u/s!Ar-uYoG1mfiLkx97L7j4MrVUMazO?e=8SSjgt) and put the unzipped files under this directory. 
Then, run run.py to obtain the results. 

Required packages can be found in requirement.txt. 

Please leave a comment or email to zengweixin13@nudt.edu.cn if you have any question

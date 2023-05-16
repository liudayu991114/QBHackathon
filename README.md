# QuantumBlack Hackathon
by Yifan **WANG**, Dayu **LIU**, Yangfan **ZHANGLIN**, Yaqi **CHEN**, Peizhen **CHEN**, Xiangying **CHEN**

## Introduction

This is the GitHub repository of team Adeptus Mechanicus for QuantumBlack Hackathon

Our team innovates at all levels of the project, including advanced image preprocessing, ensemble processing of multiple model prediction results and application of genetic algorithms.

The final performance of the models in the test set is:

| model | average AUC | highest AUC |
| :---: | :---: | :---: |
| ResNet34 | 97% | 98% |
| ResNet18 | 97% | 98% |
| ResNet50 | 96% | 97% |
| AlexNet | 94% | 95% |
| MobileNet | 94% | 95% |
| Ensemble | 99% | 99% |

## Instruction

1. First, unzip the dataset to the `data` folder
2. If you want the full experience：
    - run `pipeline.ipynb` step by step and reproduce our process and results.
3. If you want to see the results and the application directly:
  - go to https://drive.google.com/drive/folders/1VfoqsqzDW6W9JN2m_5XrDepZUcFHmLLi?usp=sharing
  - download the pretrained weights and move the `.pth` file to the `models` folder.
4. Run the deployed application by `streamlit run app.py`.
5. Enjoy our work!

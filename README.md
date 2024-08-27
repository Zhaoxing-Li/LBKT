# MLFBK: A Multi-Features with Latent Relations BERT Knowledge Tracing Model
This repository is for the research [**LBKT: A LSTM BERT-Based Knowledge Tracing Model for Long-Sequence Data.**](https://link.springer.com/chapter/10.1007/978-3-031-63031-6_15)

Full paper available at [here](https://eprints.soton.ac.uk/489244/1/2024ITS_CR_2_.pdf).

If you find this code useful in your research, please cite：

```
@inproceedings{li2024lbkt,
  title={Lbkt: a lstm bert-based knowledge tracing model for long-sequence data},
  author={Li, Zhaoxing and Yang, Jujie and Wang, Jindi and Shi, Lei and Feng, Jiayi and Stein, Sebastian},
  booktitle={International Conference on Intelligent Tutoring Systems},
  pages={174--184},
  year={2024},
  organization={Springer}
}
```

## Abstract
The field of Knowledge Tracing (KT) aims to understand how students learn and master knowledge over time by analyzing their historical behaviour data. To achieve this goal, many researchers have proposed KT models that use data from Intelligent Tutoring Systems (ITS) to predict students’ subsequent actions. However, with the development of ITS, large-scale datasets containing long-sequence data began to emerge. Recent deep learning based KT models face obstacles such as low efficiency, low accuracy, and low interpretability when dealing with large-scale datasets containing long-sequence data. To address these issues and promote the sustainable development of ITS, we propose a LSTM BERT-based Knowledge Tracing model for long sequence data processing, namely LBKT, which uses a BERT-based architecture with a Rasch model-based embeddings block to deal with different difficulty levels information and an LSTM block to process the sequential characteristic in students’ actions. LBKT achieves the best performance on most benchmark datasets on the metrics of ACC and AUC.

## Errata
If you have any question or find error in the code, you can send me a mail.

Contact: Zhaoxing Li (zhaoxing.li@soton.ac.uk).

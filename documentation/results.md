# Results

## Single Model Results

| Validation Set | Case | Duster | Master 1024 | Master 2048 | Master 4096 | RoMA 1024 | RoMA 2048 | RoMA 4096 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Focal Length** | &le;200 mm | 52.9 | 95.1 | **96.3** | 95.1 | 86.6 | 89.0 | 93.9 |
| **Focal Length** | 200-400 mm | 61.8 | 96.4 | 96.4 | **98.2** | 90.9 | 96.4 | **98.2** |
| **Focal Length** | 400-800 mm | 58.9 | 92.9 | 94.6 | 94.6 | 91.1 | **98.2** | 96.4 |
| **Focal Length** | &gt;800 | 90.2 | 100.0 | 100.0 | 100.0 | 96.1 | 98.0 | 100.0 |
| | | | | | | | | |
| **Tilt** | &lt;40&deg; | 66.3 | 96.2 | **97.3** | **97.3** | 92.4 | 95.1 | 96.7 |
| **Tilt** | &ge;40&deg; | 56.7 | 95.0 | 95.0 | 95.0 | 85.0 | 93.3 | **96.7** |
| | | | | | | | | |
| **Cloud Cov.** | &lt;40% | 74.3 | 98.8 | **99.4** | 98.8 | 96.4 | 97.6 | 98.8 |
| **Cloud Cov.** | &ge;40% | 41.6 | 89.6 | 90.9 | **92.2** | 77.9 | 88.3 | **92.2** |
| **Total (244/268)** | | 63.9 | 95.9 | **96.7** | **96.7** | 90.6 | 94.7 | **96.7** |
| | | | | | | | | |
| T<sub>in</sub> | | 34592 | 775 | 1159 | 1220 | 751 | 339 | 843 |
| Query rate (q/s) | | 36.2 | 61.7 | 61.8 | 61.8 | 42.4 | 42.8 | 42.9 |

**Caption:** Performance comparison between Duster, Master, and RoMA models. 1024, 2048, 4096 represent the maximal number of keypoints. Bold indicates best performance per row. Q/s refers to average time it took to locate a query. T<sub>in</sub>  refers to the inliers threshold. 



| Validation Set | Case | Xfeat-steerers 1024 | Xfeat-steerers 2048 | Xfeat-steerers 4096 | Xfeat* 1024 | Xfeat* 2048 | Xfeat* 4096 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Focal Length** | &le;200 mm | **76.8** | **89.0** | **81.7** | 65.9 | 76.8 | 80.5 |
| **Focal Length** | 200-400 mm | **72.7** | **85.5** | **83.6** | 69.1 | 74.5 | 76.4 |
| **Focal Length** | 400-800 mm | **75.0** | **78.6** | **80.4** | 66.1 | 75.0 | **80.4** |
| **Focal Length** | &gt;800 | **86.3** | **90.2** | **90.2** | 84.3 | 88.2 | **90.2** |
| | | | | | | | |
| **Tilt** | &lt;40&deg; | **79.3** | **87.0** | **84.8** | 71.2 | 79.9 | 84.2 |
| **Tilt** | &ge;40&deg; | **71.7** | **83.3** | **80.0** | 68.3 | 73.3 | 73.3 |
| | | | | | | | |
| **Cloud Cov.** | &lt;40% | **89.8** | **92.2** | **91.6** | 80.2 | 86.2 | 87.4 |
| **Cloud Cov.** | &ge;40% | **50.6** | **72.7** | 66.2 | 49.4 | 61.0 | **68.8** |
| **Total (244/268)** | | **77.5** | **86.1** | **83.6** | 70.5 | 78.3 | 81.6 |
| | | | | | | | |
| T<sub>in</sub> (s) | | 40.5 | 47.2 | 87.1 | 27.9 | 27.1 | 34.7 |
| Query rate (q/s) | | 6.0 | 5.2 | 2.8 | 5.24 | 5.19 | 5.21 |

**Caption:** Performance comparison of the Xfeat-steerers and Xfeat* models across different validation cases and maximal keypoint configurations. Bold indicates the best performance per column pair. Q/s refers to average time it took to locate a query.  T<sub>in</sub>  refers to the inliers threshold. 


## Ensemble Model Results



| Validation Set | Case | Ensemble Method | 1024 | 2048 | 4096 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Focal Length** | &le;200 mm | Master + RoMA  | **96.3%** | **96.3%** | 93.9% |
| **Focal Length** | 200-400 mm | Master + RoMA  | **96.4%** | **96.4%** | **96.4%** |
| **Focal Length** | 400-800 mm | Master + RoMA | **96.4%** | **96.4%** | 94.6% |
| **Focal Length** | &gt;800 mm | Master + RoMA  | **100.0%**| **100.0%**| **100.0%**|
| | | | | | |
| **Tilt** | &lt;40&deg; | Master + RoMA  | **97.8%** | **97.8%** | 97.3% |
| **Tilt** | &ge;40&deg; | Master + RoMA  | **95.0%** | **95.0%** | 91.7% |
| | | | | | |
| **Cloud Cov.** | &lt;40% | Master + RoMA | **99.4%** | **99.4%** | 98.8% |
| **Cloud Cov.** | &ge;40% | Master + RoMA | **92.2%** | **92.2%** | 89.6% |
| **Total** | | Master + RoMA  | **97.1%** | **97.1%** | 95.9% |
| | | | | | |
| Query/Second | | Master + RoMA  | 57.38 | 58.84 | 58.79 |

**Caption:** Soft ensemble performance for Master + RoMA. Columns indicate the maximum number of keypoints. Bold indicates the best performance per row.





| Validation Set | Case | Ensemble Method | 1024 | 2048 | 4096 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Focal Length** | &le;200 mm | Xfeet*-steerers + RoMA  | **96.3%** | **96.3%** | 95.1% |
| **Focal Length** | 200-400 mm | Xfeet*-steerers + RoMA  | **98.2%** | 96.4% | 90.9% |
| **Focal Length** | 400-800 mm | Xfeet*-steerers + RoMA  | **96.4%** | 89.3% | 89.3% |
| **Focal Length** | &gt;800 mm | Xfeet*-steerers + RoMA  | **98.0%** | 86.3% | 94.1% |
| | | | | | |
| **Tilt** | &lt;40&deg; | Xfeet*-steerers + RoMA | **97.3%** | 92.4% | 93.5% |
| **Tilt** | &ge;40&deg; | Xfeet*-steerers + RoMA | **96.7%** | 93.3% | 90.0% |
| | | | | | |
| **Cloud Cov.** | &lt;40% | Xfeet*-steerers + RoMA  | **98.2%** | 92.8% | 92.2% |
| **Cloud Cov.** | &ge;40% | Xfeet*-steerers + RoMA  | **94.8%** | 92.2% | 93.5% |
| **Total** | | Xfeet*-steerers + RoMA  | **97.1%** | 92.6% | 92.6% |
| | | | | | |
| Query/Second | | Xfeet*-steerers + RoMA  | 24.24 | 20.7 | 20.40 |

**Caption:** Soft ensemble performance for Xfeet*-steerers + RoMA. Columns indicate the maximum number of keypoints. Bold indicates the best performance per row. Bold indicates the best performance per column pair. Q/s refers to average time it took to locate a query.

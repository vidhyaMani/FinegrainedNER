# Final Model Comparison

Ranking by F1 score, with intrinsic, retrieval, and efficiency metrics.

|   Rank | Model       |   Precision |   Recall |     F1 |   nDCG@10 |   MRR@10 |   Params (M) |   Latency (ms) |
|-------:|:------------|------------:|---------:|-------:|----------:|---------:|-------------:|---------------:|
|      1 | bert_ner    |      0.5544 |   0.4718 | 0.5098 |    0.4795 |   0.4404 |       108.9  |          32.8  |
|      2 | roberta_ner |      0.508  |   0.4749 | 0.4909 |    0.4851 |   0.4459 |       124.07 |          33.67 |
|      3 | bilstm_crf  |      0.6091 |   0.353  | 0.4469 |    0.4822 |   0.4435 |         5.06 |           4.21 |
|      4 | cnn_bilstm  |      0.663  |   0.2922 | 0.4056 |    0.4818 |   0.4408 |         4.98 |           3.91 |
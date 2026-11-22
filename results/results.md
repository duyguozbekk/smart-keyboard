| Model | Precision | Word PPL ↓ | Top-3 acc ↑ | Size (MB) | Latency (ms, mean) |
|---|---|---|---|---|---|
| Trigram (add-one) | float64 counts | 415126.6 | 22.0% | 35.9 | 2.1 |
| Trigram (Kneser-Ney) | float64 counts | 2940.7 | 29.0% | 42.7 | 2.7 |
| Feed-forward NN LM | FP32 | 8316.3 | 26.5% | 20.8 | 8.2 |
| GPT-2 fine-tuned | FP32 | 89.2 | 48.8% | 474.7 | 155.4 |
| GPT-2 fine-tuned INT8 | INT8 dynamic (blocks), FP32 head | 108.3 | 47.0% | 231.8 | 166.1 |

Test set: WikiText-2 test, 201,560 words; top-3 accuracy on 1000 positions; latency averaged over 1000 runs per model on Apple M2 CPU (4 threads), Darwin 27.2.0.

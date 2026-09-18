# Offline evaluation

Task: put the white mug on the left plate and put the yellow and white mug on the right plate

Training episode: 0; held-out episode: 18.

Both metrics are lower-is-better; padding is excluded and means are weighted by valid actions.

| Model | Split | Samples | Flow loss (t=0.5) | Normalized action MSE |
| --- | --- | --- | --- | --- |
| base | train | 3 | 1.760037 | 1.116051 |
| base | heldout | 3 | 1.266111 | 1.073904 |
| finetuned | train | 3 | 0.969049 | 0.662704 |
| finetuned | heldout | 3 | 0.939694 | 0.649092 |

## Limitations

- Small deterministic frame sample from one episode per split, one noise draw and t=0.5.
- Held out from this fine-tuning run only; base pretraining overlap is not established.
- Both models reuse saved dataset-global normalization statistics, not train-only statistics.
- Action MSE measures agreement with one demonstration, not task success. No environment rollout.
- Base weights are evaluated with the same adapted 7D configuration as the fine-tuned weights.

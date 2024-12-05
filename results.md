# Results

Some results tabulated for easier digestion/comparison. More visualizations can be found on the [interactive benchmark page](https://subnet.chunking.com/benchmarks/c8dfa00c-f21d-4233-8491-b6396946dca4)

- Recall measures the percentage of all available relevant context that was included in the chunk. We used [Raga's LLM Based Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/) metric.
- Precision measures the percentage of relevant context only within the chunk itself. We used [Raga's LLM Based Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/) metric.
- F1-score is the harmonic mean of recall and precision. It's commonly used as a way to integrate both metrics.

## All Corpora

### Token Limit: 50

| Chunker      | Recall | Precision |
| ------------ | ------ | --------- |
| ai21         | 0.388  | 0.581     |
| miner177     | 0.325  | 0.455     |
| miner23      | 0.238  | 0.341     |
| miner5       | 0.438  | 0.692     |
| Old Subnet   | 0.236  | 0.382     |
| unstructured | 0.419  | 0.630     |

### Token Limit: 150

| Chunker      | Recall | Precision |
| ------------ | ------ | --------- |
| ai21         | 0.719  | 0.843     |
| miner177     | 0.611  | 0.797     |
| miner23      | 0.435  | 0.563     |
| miner5       | 0.680  | 0.882     |
| Old Subnet   | 0.401  | 0.568     |
| unstructured | 0.726  | 0.904     |

### Token Limit: 250

| Chunker      | Recall | Precision |
| ------------ | ------ | --------- |
| ai21         | 0.824  | 0.910     |
| miner177     | 0.655  | 0.811     |
| miner23      | 0.597  | 0.732     |
| miner5       | 0.775  | 0.908     |
| Old Subnet   | 0.548  | 0.709     |
| unstructured | 0.818  | 0.920     |

### F1-scores by token limit

| Token Limit | Miner 5 | Unstructured | AI21  |
| ----------- | ------- | ------------ | ----- |
| 50          | 0.536   | 0.503        | 0.464 |
| 150         | 0.768   | 0.805        | 0.776 |
| 250         | 0.836   | 0.866        | 0.864 |

### F1-scores by token limit for top miners (in descending order of rank in the subnet, going from left to right)

| Token Limit | 5     | 177   | 23    |
| ----------- | ----- | ----- | ----- |
| 150         | 0.536 | 0.379 | 0.280 |
| 250         | 0.768 | 0.692 | 0.491 |
| 500         | 0.836 | 0.724 | 0.658 |

### Top miner's performance versus the better of Unstructured or AI21

| Token Limit | Recall | Precision |
| ----------- | ------ | --------- |
| 50          | +4.47% | +9.81%    |
| 150         | -6.37% | -2.41%    |
| 250         | -5.98% | -1.34%    |

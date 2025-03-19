# Li-Paper Analysis Summary

Analysis Date: 2025-03-20 01:52:30

## Overview

- **Number of Papers Analyzed**: 30
- **Total Report Files Processed**: 60

## Agreement Rates by Configuration

| Configuration | Mean Agreement Rate (%) | Std Dev | Count |
|---------------|-------------------------|---------|-------|
| claude_openai | 78.19047619047619 | 20.403851292064314 | 30 |
| openai_claude | 95.23809523809523 | 6.6408131113743725 | 30 |

## Statistical Comparison

- **Mann-Whitney U Test**: U = 668.5, p-value = 0.001049906476217263
- **Significant Difference at α=0.05**: Yes

## Model Output Agreement

- **Overall Model Output Agreement Rate**: 44.76%
- **Correlation with OpenAI-Claude Validator Agreement**: -0.2133
- **Correlation with Claude-OpenAI Validator Agreement**: -0.4563

## Top 5 Papers by Model Output Agreement

| Paper ID | Model Output Agreement Rate (%) |
|----------|--------------------------------|
| 34338412 | 94.28571428571428 |
| 32591734 | 82.85714285714286 |
| 20473188 | 77.14285714285715 |
| 33450821 | 62.857142857142854 |
| 23811246 | 60.0 |

## Bottom 5 Papers by Model Output Agreement

| Paper ID | Model Output Agreement Rate (%) |
|----------|--------------------------------|
| 29894495 | 25.71428571428571 |
| 32688083 | 22.857142857142858 |
| 29305022 | 22.857142857142858 |
| 32925614 | 17.142857142857142 |
| 32069337 | 5.714285714285714 |

## Visualizations

The following visualizations have been generated:

1. `agreement_by_config.png`: Bar chart showing mean agreement rates by configuration
2. `agreement_distribution.png`: Violin plot showing the distribution of agreement rates
3. `agreement_by_item.png`: Bar chart showing agreement rates by checklist item
4. `model_output_agreement_by_paper.png`: Bar chart showing model output agreement by paper
5. `model_output_agreement_by_item.png`: Bar chart showing model output agreement by checklist item
6. `correlation_plots.png`: Scatter plots showing correlations between validator agreement and model output agreement


# RRWP Universal Sweep — Analysis Report

**Date:** 2026-03-27
**Total configurations evaluated:** 333
**Datasets:** QM9, ZINC, PubChem16, PubChem32, PubChem64
**Parameter grid:** dims ∈ [512, 768, 1024], depths ∈ [2, 3], bins ∈ [6, 8], k_values ∈ ['4', '6,12', '6,24', '4,8,12', '6,10,14', '6,12,16', '4,8,12,16']
**Fixed:** decoder=greedy, beam_size=32, seed=42

---

## 1. Dataset Difficulty Overview

| Dataset | Accuracy Range | Mean | Std | Verdict |
|---------|---------------|------|-----|---------|
| QM9 | 0.988 – 0.994 | 0.992 | 0.002 | Near-ceiling, trivial |
| ZINC | 0.868 – 0.989 | 0.967 | 0.027 | High, config-sensitive |
| PubChem16 | 0.978 – 0.992 | 0.987 | 0.004 | High, config-sensitive |
| PubChem32 | 0.872 – 0.936 | 0.913 | 0.015 | Moderate, config matters |
| PubChem64 | 0.800 – 0.884 | 0.857 | 0.019 | Challenging, config critical |

> **QM9** achieves >98.8% for *all* configurations. It is not discriminative and is excluded from configuration selection.

## 2. Parameter Importance (eta-squared, ANOVA)

Eta-squared (η²) measures the proportion of variance in graph accuracy explained by each parameter.
Higher η² = more influential parameter.

| Parameter | ZINC η² | PubChem16 η² | PubChem32 η² | PubChem64 η² | Avg η² | Rank |
|-----------|---------|-------------|-------------|-------------|--------|------|
| **K Values** | 0.661 | 0.282 | 0.253 | 0.224 | 0.355 | 1 |
| **Depth** | 0.098 | 0.193 | 0.382 | 0.342 | 0.254 | 2 |
| **Bins** | 0.080 | 0.082 | 0.219 | 0.217 | 0.150 | 3 |
| **Dimension** | 0.003 | 0.006 | 0.022 | 0.043 | 0.018 | 4 |

## 3. Key Findings by Parameter

### 3.1 Bins (6 vs 8)

- **ZINC:** bins=6 → 0.9594, bins=8 → 0.9746 (Δ = +0.0152)
- **PubChem16:** bins=6 → 0.9861, bins=8 → 0.9881 (Δ = +0.0020)
- **PubChem32:** bins=6 → 0.9059, bins=8 → 0.9202 (Δ = +0.0142)
- **PubChem64:** bins=6 → 0.8477, bins=8 → 0.8655 (Δ = +0.0178)

> **bins=8 is universally better** across all hard datasets. More bins = finer RRWP discretization = better node disambiguation.

### 3.2 Dimension (512 vs 768 vs 1024)

- **ZINC** dim=512: 0.9649
- **ZINC** dim=768: 0.9684
- **ZINC** dim=1024: 0.9665
- **PubChem16** dim=512: 0.9872
- **PubChem16** dim=768: 0.9867
- **PubChem16** dim=1024: 0.9873
- **PubChem32** dim=512: 0.9100
- **PubChem32** dim=768: 0.9135
- **PubChem32** dim=1024: 0.9154
- **PubChem64** dim=512: 0.8510
- **PubChem64** dim=768: 0.8590
- **PubChem64** dim=1024: 0.8597

> **Dimension is the least impactful parameter** (η²=0.018). ZINC: 512→768 gains +0.4pp, 768→1024 gains -0.2pp.
> Even on PubChem64 (hardest), the dim effect is only ~1pp across the full range. Other parameters matter far more.

### 3.3 K Values (random walk step lengths)

The k_values parameter defines which random walk step lengths are used as RRWP features.

**ZINC:**
  - k=6,12: mean=0.9691, max=0.988
  - k=6,24: mean=0.9220, max=0.961
  - k=4,8,12: mean=0.9795, max=0.986
  - k=6,10,14: mean=0.9769, max=0.988
  - k=6,12,16: mean=0.9807, max=0.989
  - k=4,8,12,16: mean=0.9805, max=0.985

**PubChem16:**
  - k=6,12: mean=0.9848, max=0.992
  - k=4,8,12: mean=0.9868, max=0.990
  - k=6,10,14: mean=0.9878, max=0.992
  - k=6,12,16: mean=0.9857, max=0.990
  - k=4,8,12,16: mean=0.9902, max=0.992

**PubChem32:**
  - k=6,12: mean=0.8992, max=0.920
  - k=4,8,12: mean=0.9163, max=0.930
  - k=6,10,14: mean=0.9149, max=0.930
  - k=6,12,16: mean=0.9132, max=0.930
  - k=4,8,12,16: mean=0.9228, max=0.936

**PubChem64:**
  - k=6,12: mean=0.8400, max=0.872
  - k=4,8,12: mean=0.8587, max=0.876
  - k=6,10,14: mean=0.8568, max=0.884
  - k=6,12,16: mean=0.8598, max=0.872
  - k=4,8,12,16: mean=0.8675, max=0.884

> **More k features generally help** (4 features > 3 > 2). `k={4,8,12,16}` (4 features) and `k={6,12,16}` (3 features, wider) consistently rank at top. The `k={6,24}` pair (extreme spacing) is an outlier — very low for ZINC.

### 3.4 Depth (2 vs 3)

- **ZINC:** depth=2 → 0.9582, depth=3 → 0.9749 (Δ = +0.0168)
- **PubChem16:** depth=2 → 0.9855, depth=3 → 0.9886 (Δ = +0.0031)
- **PubChem32:** depth=2 → 0.9030, depth=3 → 0.9219 (Δ = +0.0189)
- **PubChem64:** depth=2 → 0.8454, depth=3 → 0.8677 (Δ = +0.0223)

> **Depth is the #2 most impactful parameter** after k_values. depth=3 consistently outperforms depth=2 by 1.5–2.2pp on the hard datasets. This is a meaningful gain — always prefer depth=3.

## 4. Top Configurations (Recommended)

Ranked by harmonic mean of graph accuracy across ZINC, PubChem16, PubChem32, PubChem64.
Harmonic mean penalizes configurations that are weak on any single dataset.

### Rank 1: `1024_d3_k6,10,14_b8`

| Parameter | Value |
|-----------|-------|
| **Dimension** | 1024 |
| **Depth** | 3 |
| **K Values** | 6,10,14 |
| **Bins** | 8 |
| **Harmonic Mean** | 0.9464 |

| Dataset | Accuracy |
|---------|----------|
| QM9 | 0.993 |
| ZINC | 0.988 |
| PubChem16 | 0.992 |
| PubChem32 | 0.930 |
| PubChem64 | 0.884 |

### Rank 2: `768_d3_k6,10,14_b8`

| Parameter | Value |
|-----------|-------|
| **Dimension** | 768 |
| **Depth** | 3 |
| **K Values** | 6,10,14 |
| **Bins** | 8 |
| **Harmonic Mean** | 0.9441 |

| Dataset | Accuracy |
|---------|----------|
| QM9 | 0.993 |
| ZINC | 0.987 |
| PubChem16 | 0.990 |
| PubChem32 | 0.924 |
| PubChem64 | 0.884 |

### Rank 3: `1024_d3_k6,12,16_b8`

| Parameter | Value |
|-----------|-------|
| **Dimension** | 1024 |
| **Depth** | 3 |
| **K Values** | 6,12,16 |
| **Bins** | 8 |
| **Harmonic Mean** | 0.9421 |

| Dataset | Accuracy |
|---------|----------|
| QM9 | 0.993 |
| ZINC | 0.989 |
| PubChem16 | 0.990 |
| PubChem32 | 0.930 |
| PubChem64 | 0.870 |

## 5. Compact Configuration (dim=512)

Best dim=512 config: **`512_d3_k4,8,12_b8`**

| Parameter | Value |
|-----------|-------|
| **Dimension** | 512 |
| **Depth** | 3 |
| **K Values** | 4,8,12 |
| **Bins** | 8 |
| **Harmonic Mean** | 0.9414 |

| Dataset | Accuracy |
|---------|----------|
| ZINC | 0.985 |
| PubChem16 | 0.988 |
| PubChem32 | 0.928 |
| PubChem64 | 0.874 |

> Compact config achieves H-mean=0.9414 vs best=0.9464 (Δ = 0.0050).
> This is a 0.5% relative drop for a 512-dim reduction.

## 6. Final Recommendations for Generation Pipeline

Based on the analysis, we recommend testing the following configurations in the generation pipeline:

| Config | Dim | Depth | K Values | Bins | Purpose |
|--------|-----|-------|----------|------|---------|
| **Best** | 1024 | 3 | 6,10,14 | 8 | Maximum accuracy across all datasets |
| **Compact** | 512 | 3 | 4,8,12 | 8 | Smaller embedding, faster training |
| **Runner-up** | 768 | 3 | 6,10,14 | 8 | Alternative top config |

### Key Takeaways

1. **bins=8 > bins=6**: Universally. Always use 8 bins.
2. **More k features help**: 4 features (`4,8,12,16`) is best, 3 features (`6,12,16`) is a close second.
3. **Depth=3 is worth it**: depth=3 consistently gains 1.5–2.2pp over depth=2 on hard datasets (η²=0.254, rank #2). Always use depth=3.
4. **Dimension barely matters**: dim has the smallest effect (η²=0.018). 512 vs 1024 differs by <1pp on most datasets, so 512 is viable for compact models.
5. **QM9 is solved**: All configs achieve >98.8%. Focus tuning efforts on ZINC and PubChem.
6. **PubChem64 is the stress test**: Accuracy ranges from 80.0% to 88.4%. This dataset most rewards the right configuration.

# QQA Parameter Comparison: QQA1-QQA10 vs Baseline QQA

## Baseline QQA Configuration
```
sol_size:        100
learning_rate:   1.0
temp:            0.001
min_bg:          -3.0
max_bg:          0.1
curve_rate:      4.0
div_param:       0.2
num_epochs:      3000
check_interval:  500
plot_dynamics:   false
device:          null
```

---

## Systematic Parameter Comparison

### QQA1 vs QQA
| Parameter       | QQA (Baseline) | QQA1      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | **256**   | **+156 (+156%)**         |
| learning_rate   | 1.0            | **0.75**  | **-0.25 (-25%)**         |
| temp            | 0.001          | **0.0005**| **-0.0005 (-50%)**       |
| min_bg          | -3.0           | -3.0      | **Same**                 |
| max_bg          | 0.1            | **0.05**  | **-0.05 (-50%)**         |
| curve_rate      | 4.0            | 4.0       | **Same**                 |
| div_param       | 0.2            | **0.25**  | **+0.05 (+25%)**         |
| num_epochs      | 3000           | **3500**  | **+500 (+16.7%)**        |
| check_interval  | 500            | 500       | **Same**                 |

**Key Changes:** Larger batch size (256), lower learning rate (0.75), lower temperature (0.0005), lower max background field (0.05), slightly higher diversity (0.25), more epochs (3500).

---

### QQA2 vs QQA
| Parameter       | QQA (Baseline) | QQA2      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | **128**   | **+28 (+28%)**           |
| learning_rate   | 1.0            | **1.2**   | **+0.2 (+20%)**          |
| temp            | 0.001          | **0.01**  | **+0.009 (+900%)**       |
| min_bg          | -3.0           | **-2.5**  | **+0.5 (+16.7%)**        |
| max_bg          | 0.1            | **0.05**  | **-0.05 (-50%)**         |
| curve_rate      | 4.0            | **4**     | **Same (integer)**       |
| div_param       | 0.2            | **0.18**  | **-0.02 (-10%)**         |
| num_epochs      | 3000           | **2500**  | **-500 (-16.7%)**        |
| check_interval  | 500            | **400**   | **-100 (-20%)**          |

**Key Changes:** Larger batch size (128), higher learning rate (1.2), **much higher temperature (0.01, 10x)**, less negative min background (-2.5), lower max background (0.05), fewer epochs (2500).

---

### QQA3 vs QQA
| Parameter       | QQA (Baseline) | QQA3      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | 100       | **Same**                 |
| learning_rate   | 1.0            | **0.9**   | **-0.1 (-10%)**          |
| temp            | 0.001          | **0.0005**| **-0.0005 (-50%)**       |
| min_bg          | -3.0           | **-4.0**  | **-1.0 (-33.3%)**        |
| max_bg          | 0.1            | **0.0**   | **-0.1 (-100%)**         |
| curve_rate      | 4.0            | **6.0**   | **+2.0 (+50%)**          |
| div_param       | 0.2            | **0.15**  | **-0.05 (-25%)**         |
| num_epochs      | 3000           | **3500**  | **+500 (+16.7%)**        |
| check_interval  | 500            | 500       | **Same**                 |

**Key Changes:** Lower learning rate (0.9), lower temperature (0.0005), **more negative min background (-4.0)**, **zero max background (0.0)**, **higher curve rate (6.0)**, lower diversity (0.15), more epochs (3500).

---

### QQA4 vs QQA
| Parameter       | QQA (Baseline) | QQA4      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | **200**   | **+100 (+100%)**         |
| learning_rate   | 1.0            | 1.0       | **Same**                 |
| temp            | 0.001          | **0.002** | **+0.001 (+100%)**       |
| min_bg          | -3.0           | -3.0      | **Same**                 |
| max_bg          | 0.1            | 0.1       | **Same**                 |
| curve_rate      | 4.0            | 4.0       | **Same**                 |
| div_param       | 0.2            | **0.40**  | **+0.2 (+100%)**         |
| num_epochs      | 3000           | **2500**  | **-500 (-16.7%)**        |
| check_interval  | 500            | **400**   | **-100 (-20%)**          |

**Key Changes:** **Larger batch size (200, 2x)**, **higher temperature (0.002, 2x)**, **much higher diversity (0.40, 2x)**, fewer epochs (2500).

---

### QQA5 vs QQA
| Parameter       | QQA (Baseline) | QQA5      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | **128**   | **+28 (+28%)**           |
| learning_rate   | 1.0            | **0.8**   | **-0.2 (-20%)**          |
| temp            | 0.001          | 0.001     | **Same**                 |
| min_bg          | -3.0           | **-3.5**  | **-0.5 (-16.7%)**        |
| max_bg          | 0.1            | **0.05**  | **-0.05 (-50%)**         |
| curve_rate      | 4.0            | **5**     | **+1.0 (+25%)**          |
| div_param       | 0.2            | **0.25**  | **+0.05 (+25%)**         |
| num_epochs      | 3000           | **6000**  | **+3000 (+100%)**        |
| check_interval  | 500            | **750**   | **+250 (+50%)**          |

**Key Changes:** Larger batch size (128), lower learning rate (0.8), more negative min background (-3.5), lower max background (0.05), higher curve rate (5), **double the epochs (6000)**, longer check interval (750).

---

### QQA6 vs QQA
| Parameter       | QQA (Baseline) | QQA6      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | 100       | **Same**                 |
| learning_rate   | 1.0            | 1.0       | **Same**                 |
| temp            | 0.001          | 0.001     | **Same**                 |
| min_bg          | -3.0           | **-2**    | **+1.0 (+33.3%)**        |
| max_bg          | 0.1            | 0.1       | **Same**                 |
| curve_rate      | 4.0            | **4**     | **Same (integer)**       |
| div_param       | 0.2            | 0.2       | **Same**                 |
| num_epochs      | 3000           | 3000      | **Same**                 |
| check_interval  | 500            | 500       | **Same**                 |

**Key Changes:** **Only min_bg differs: -2.0 instead of -3.0** (less negative). Comments indicate this follows paper's general γ_min = -2 setting.

---

### QQA7 vs QQA
| Parameter       | QQA (Baseline) | QQA7      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | 100       | **Same**                 |
| learning_rate   | 1.0            | **0.1**   | **-0.9 (-90%)**          |
| temp            | 0.001          | 0.001     | **Same**                 |
| min_bg          | -3.0           | **-2**    | **+1.0 (+33.3%)**        |
| max_bg          | 0.1            | 0.1       | **Same**                 |
| curve_rate      | 4.0            | **4**     | **Same (integer)**       |
| div_param       | 0.2            | 0.2       | **Same**                 |
| num_epochs      | 3000           | 3000      | **Same**                 |
| check_interval  | 500            | 500       | **Same**                 |

**Key Changes:** **Much lower learning rate (0.1, 10x reduction)**, min_bg set to -2 (paper's general setting). Tests lower learning rate from paper's exploration.

---

### QQA8 vs QQA
| Parameter       | QQA (Baseline) | QQA8      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | 100       | **Same**                 |
| learning_rate   | 1.0            | **0.01**  | **-0.99 (-99%)**         |
| temp            | 0.001          | 0.001     | **Same**                 |
| min_bg          | -3.0           | **-2**    | **+1.0 (+33.3%)**        |
| max_bg          | 0.1            | 0.1       | **Same**                 |
| curve_rate      | 4.0            | **4**     | **Same (integer)**       |
| div_param       | 0.2            | 0.2       | **Same**                 |
| num_epochs      | 3000           | 3000      | **Same**                 |
| check_interval  | 500            | 500       | **Same**                 |

**Key Changes:** **Much lower learning rate (0.01, 100x reduction)**, min_bg set to -2 (paper's general setting). Tests lowest learning rate from paper.

---

### QQA9 vs QQA
| Parameter       | QQA (Baseline) | QQA9      | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | 100       | **Same**                 |
| learning_rate   | 1.0            | 1.0       | **Same**                 |
| temp            | 0.001          | 0.001     | **Same**                 |
| min_bg          | -3.0           | **-20**   | **-17.0 (-566.7%)**      |
| max_bg          | 0.1            | 0.1       | **Same**                 |
| curve_rate      | 4.0            | **4**     | **Same (integer)**       |
| div_param       | 0.2            | 0.2       | **Same**                 |
| num_epochs      | 3000           | 3000      | **Same**                 |
| check_interval  | 500            | 500       | **Same**                 |

**Key Changes:** **Much more negative min_bg: -20** (paper's setting for Max Cut large graphs). All other parameters match baseline.

---

### QQA10 vs QQA
| Parameter       | QQA (Baseline) | QQA10     | Difference               |
|-----------------|----------------|-----------|--------------------------|
| sol_size        | 100            | 100       | **Same**                 |
| learning_rate   | 1.0            | 1.0       | **Same**                 |
| temp            | 0.001          | 0.001     | **Same**                 |
| min_bg          | -3.0           | **-5**    | **-2.0 (-66.7%)**        |
| max_bg          | 0.1            | 0.1       | **Same**                 |
| curve_rate      | 4.0            | **4**     | **Same (integer)**       |
| div_param       | 0.2            | 0.2       | **Same**                 |
| num_epochs      | 3000           | 3000      | **Same**                 |
| check_interval  | 500            | 500       | **Same**                 |

**Key Changes:** **More negative min_bg: -5** (paper's setting for Max Cut other graphs). All other parameters match baseline.

---

## Summary by Parameter Category

### sol_size (Batch Size)
- **Same as baseline:** QQA, QQA3, QQA6-QQA10 (all 100)
- **Increased:** QQA1 (256, +156%), QQA4 (200, +100%), QQA2 (128, +28%), QQA5 (128, +28%)

### learning_rate
- **Same as baseline:** QQA, QQA4, QQA6, QQA9, QQA10 (all 1.0)
- **Increased:** QQA2 (1.2, +20%)
- **Decreased:** QQA7 (0.1, -90%), QQA8 (0.01, -99%), QQA1 (0.75, -25%), QQA3 (0.9, -10%), QQA5 (0.8, -20%)

### temp (Temperature)
- **Same as baseline:** QQA, QQA5, QQA6-QQA10 (all 0.001)
- **Increased:** QQA2 (0.01, +900%), QQA4 (0.002, +100%)
- **Decreased:** QQA1 (0.0005, -50%), QQA3 (0.0005, -50%)

### min_bg (Minimum Background Field)
- **Same as baseline:** QQA, QQA1, QQA4 (-3.0)
- **Less negative (closer to 0):** QQA2 (-2.5), QQA6-QQA8 (-2.0)
- **More negative:** QQA3 (-4.0), QQA5 (-3.5), QQA10 (-5.0), QQA9 (-20.0)

### max_bg (Maximum Background Field)
- **Same as baseline:** QQA, QQA4, QQA6-QQA10 (all 0.1)
- **Decreased:** QQA1 (0.05, -50%), QQA2 (0.05, -50%), QQA5 (0.05, -50%)
- **Zero:** QQA3 (0.0, -100%)

### curve_rate
- **Same as baseline:** QQA, QQA1, QQA2, QQA4, QQA6-QQA10 (all 4)
- **Increased:** QQA3 (6.0, +50%), QQA5 (5, +25%)

### div_param (Diversity Parameter)
- **Same as baseline:** QQA, QQA6-QQA10 (all 0.2)
- **Increased:** QQA1 (0.25, +25%), QQA4 (0.40, +100%), QQA5 (0.25, +25%)
- **Decreased:** QQA2 (0.18, -10%), QQA3 (0.15, -25%)

### num_epochs
- **Same as baseline:** QQA, QQA6-QQA10 (all 3000)
- **Increased:** QQA1 (3500, +16.7%), QQA3 (3500, +16.7%), QQA5 (6000, +100%)
- **Decreased:** QQA2 (2500, -16.7%), QQA4 (2500, -16.7%)

### check_interval
- **Same as baseline:** QQA, QQA1, QQA3, QQA6-QQA10 (all 500)
- **Increased:** QQA5 (750, +50%)
- **Decreased:** QQA2 (400, -20%), QQA4 (400, -20%)

---

## Configuration Groups by Purpose

### Paper-Based Configurations (QQA6-QQA10)
- **QQA6-QQA8:** Learning rate sweep (lr = 1.0, 0.1, 0.01) with paper's general γ_min = -2
- **QQA9:** Paper's setting for Max Cut large graphs (γ_min = -20)
- **QQA10:** Paper's setting for Max Cut other graphs (γ_min = -5)

### Exploration Configurations (QQA1-QQA5)
- **QQA1:** Larger batch + lower temperature + lower learning rate
- **QQA2:** Higher temperature (10x) + higher learning rate + fewer epochs
- **QQA3:** More aggressive penalty schedule (min_bg=-4, max_bg=0, curve_rate=6)
- **QQA4:** High diversity (0.4) + larger batch + higher temperature
- **QQA5:** Extended schedule (6000 epochs) + balanced parameters


## Full Parameter Matrix

| Parameter      | QQA  | QQA1 | QQA2 | QQA3 | QQA4 | QQA5 | QQA6 | QQA7 | QQA8 | QQA9 | QQA10 |
|----------------|------|------|------|------|------|------|------|------|------|------|-------|
| sol_size       | 100  | 256  | 128  |      | 200  | 128  |      |      |      |      |      |
| learning_rate  | 1.0  | 0.75 | 1.2  | 0.9  |      | 0.8  |      | 0.1  | 0.01 |      |      |
| temp           | 0.001| 0.0005| 0.01 | 0.0005| 0.002|      |      |      |      |      |      |
| min_bg         | -3.0 |      | -2.5 | -4.0 |      | -3.5 | -2.0 | -2.0 | -2.0 | -20.0| -5.0  |
| max_bg         | 0.1  | 0.05 | 0.05 | 0.0  |      | 0.05 |      |      |      |      |      |
| curve_rate     | 4.0  |      |      | 6.0  |      | 5.0  |      |      |      |      |      |
| div_param      | 0.2  | 0.25 | 0.18 | 0.15 | 0.40 | 0.25 |      |      |      |      |      |
| num_epochs     | 3000 | 3500 | 2500 | 3500 | 2500 | 6000 |      |      |      |      |      |
| check_interval | 500  |      | 400  |      | 400  | 750  |      |      |      |      |      |
| plot_dynamics  | false|      |      |      |      |      |      |      |      |      |       |
| device         | null |      |      |      |      |      |      |      |      |      |       |


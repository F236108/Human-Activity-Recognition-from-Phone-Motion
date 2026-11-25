# Analysis – AML Assignment 4

## Task 1.3 – Example Windows Description

In `example_windows.png`, the three moving activities (`WALKING`, `WALKING_UPSTAIRS`,
`WALKING_DOWNSTAIRS`) show strong, periodic oscillations across all three body-acceleration
and gyroscope axes driven by repeated footsteps and torso rotation. By contrast, the three
still activities (`SITTING`, `STANDING`, `LAYING`) remain nearly flat around zero in both
body acceleration and angular velocity because the waist-worn phone undergoes almost no
dynamic motion.

---

## Table 1 – Data

| Split | Windows | Volunteers | Subject IDs |
|---|---|---|---|
| Train | 5,891 | 17 | 1, 5, 6, 7, 8, 11, 14, 16, 17, 19, 21, 22, 23, 26, 27, 28, 29 |
| Validation | 1,461 | 4 | 3, 15, 25, 30 |
| Test | 2,947 | 9 | 2, 4, 9, 10, 12, 13, 18, 20, 24 |

### Channel Means and Standard Deviations (Training Windows Only)

| Channel | Mean | Std Dev |
|---|---|---|
| `body_acc_x` | -0.000702 | 0.198904 |
| `body_acc_y` | -0.000268 | 0.122580 |
| `body_acc_z` | -0.000322 | 0.108991 |
| `body_gyro_x` | -0.000759 | 0.401472 |
| `body_gyro_y` | -0.000765 | 0.389227 |
| `body_gyro_z` | -0.000292 | 0.265129 |
| `total_acc_x` | 0.799970 | 0.419442 |
| `total_acc_y` | 0.020705 | 0.383104 |
| `total_acc_z` | 0.076466 | 0.376729 |

---

## Table 2 – Network and Training

| Metric | Value |
|---|---|
| Trainable parameters | 46,982 |
| Initial loss (before training) | 1.8058 |
| Best epoch (lowest validation loss) | 3 |
| Training loss at best epoch | 0.1451 |
| Validation loss at best epoch | 0.1304 |
| Validation accuracy at best epoch | 96.44% |
| Training time | 495.8s |
| Hardware | CPU (Intel64 Family 6 Model 158 Stepping 13, GenuineIntel) |

---

## Table 3 – Test Results

| Metric | Walk | Up | Down | Sit | Stand | Lay |
|---|---|---|---|---|---|---|
| Per-class accuracy | 87.70% | 98.94% | 98.10% | 77.39% | 83.46% | 100.00% |
| Mean of max $\alpha_t$ | 0.2863 | 0.2174 | 0.2640 | 0.0166 | 0.0162 | 0.0104 |

**Test accuracy:** **90.74%** (over 2,947 windows; flat attention reference $1/128 \approx 0.0078$).

### 6×6 Test Confusion Matrix (rows = True, columns = Predicted)

| True \ Pred | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---|---|---|---|---|---|
| WALKING | 435 | 43 | 14 | 2 | 2 | 0 |
| WALKING_UPSTAIRS | 3 | 466 | 2 | 0 | 0 | 0 |
| WALKING_DOWNSTAIRS | 4 | 4 | 412 | 0 | 0 | 0 |
| SITTING | 0 | 2 | 0 | 380 | 109 | 0 |
| STANDING | 1 | 0 | 0 | 87 | 444 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 0 | 537 |

---

## Task 6 – Analysis Questions

### 6.1  Why is the validation set chosen by person rather than by picking random windows?

Adjacent windows overlap by 50% (128-sample windows slide 64 samples at a time).
If we split randomly, a window in the validation set might share 64 samples with a
window in the training set. The model could then 'remember' those samples rather than
learning to generalise, giving an over-optimistic estimate of validation performance.
Splitting by person guarantees that every window in the validation set comes from a
volunteer never seen during training, so the validation loss is a fair proxy for
out-of-distribution generalisation.

### 6.2  Which two activities are confused most often?

SITTING and STANDING are most often confused. From the Task 1.3 plots, both
activities show near-zero body acceleration and near-zero gyroscope readings.
When the phone is worn on the waist and the person is stationary, the sensor
barely moves; the only difference is a slight offset in the gravity component,
which is subtle and easily confused after standardisation.

### 6.3  Why is LAYING recognised almost perfectly?

When a person is lying down, the z-axis of the total acceleration changes
dramatically: gravity now acts along the axis that is normally perpendicular to
the floor. The total_acc_z (or total_acc_x/y) signal shows a constant offset
completely different from all other activities, making LAYING trivially separable.

### 6.4  Is attention more peaked for moving or still activities?

Attention is more peaked (higher mean-max α_t) for moving activities such as
WALKING, WALKING_UPSTAIRS, and WALKING_DOWNSTAIRS. A periodic gait produces
distinctive peaks at specific moments of the step cycle, so the model learns to
focus on those informative instants. For still activities (SITTING, STANDING,
LAYING) the signal is almost constant throughout the window, so attention
spreads more uniformly—any moment is as informative as any other.

### 6.5  Why is BiLSTM allowed here, and when would it not be?

A BiLSTM reads the window in both directions. This is valid here because we
classify a *complete, pre-recorded 2.56-second window*: the entire sequence is
available at inference time. In a real-time streaming scenario (e.g., triggering
an alarm the moment an activity begins, with no look-ahead), the backward pass
would require future samples that have not yet arrived, making it inadmissible.

### 6.6  Why does an LSTM keep information over 128 steps better than a plain RNN?

A plain RNN updates the hidden state as h_t = tanh(W x_t + U h_{t-1} + b).
The gradient of the loss with respect to h_0 is a product of 128 Jacobian
matrices; because tanh saturates, their eigenvalues are ≤1, causing the
gradient to vanish exponentially and preventing information from propagating
backwards through long sequences.

The LSTM adds a cell state c_t updated by c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t.
When the forget gate f_t ≈ 1 and the input gate i_t ≈ 0, the cell state is
copied almost unchanged across many steps, providing a near-constant gradient
highway that prevents vanishing gradients. The gating mechanism lets the network
learn *which* information to keep, update, or discard.

Gradient clipping (clip_grad_norm_ with max_norm=1.0) guards against *exploding*
gradients: if the gradient norm exceeds 1.0, all gradients are rescaled so the
total norm equals 1.0, preventing catastrophic weight updates.

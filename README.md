# NeuroDecode Lab 

## Know When Your BCI Doesn't Know

### Overview

**NeuroDecode Lab** is an offline brain-computer interface analysis toolkit designed to answer a simple but important question:

> Can a BCI system recognize when its own predictions are likely to be unreliable?

Rather than treating every classifier output as equally trustworthy, NeuroDecode Lab augments conventional EEG/ECoG decoding with **uncertainty estimation, calibration, temporal stability analysis, and selective prediction**.

The project is designed to be entirely software-based. It requires no EEG headset, VR headset, electrodes, or specialized hardware.

This makes it a natural fit for the BR41N.IO offline data-analysis track, which explicitly supports projects using recorded BCI datasets without requiring BCI hardware.

---

## Motivation

Most BCI pipelines follow a simple structure:

```text
brain signal
    ↓
preprocessing
    ↓
feature extraction
    ↓
classifier
    ↓
predicted command
```

The classifier produces an answer, but a practical BCI system also needs to know:

**How much should we trust that answer?**

Brain signals are noisy and non-stationary. Signal quality can change across time, subjects, recording sessions, electrodes, and cognitive states.

A wrong prediction with high confidence can be much more damaging than a system that recognizes uncertainty and temporarily abstains.

NeuroDecode Lab therefore expands the pipeline:

```text
brain signal
    ↓
preprocessing
    ↓
feature extraction
    ↓
classifier
    ↓
prediction probabilities
    ↓
uncertainty + stability analysis
    ↓
prediction / abstention decision
```

---

# Core Research Question

The central question is:

> **Can decoder uncertainty predict BCI classification failure?**

A secondary question is:

> **Can selective prediction improve decoding reliability by refusing predictions during uncertain signal periods?**

Instead of optimizing only conventional accuracy, the project evaluates whether uncertainty-aware decoding produces a more reliable interface.

---

# Dataset

The system will operate on one of the offline datasets supplied through the BR41N.IO data-analysis projects.

A particularly strong candidate is the **ECoG Video Watching Analysis** task.

The supplied ECoG dataset contains recordings collected while a participant watched video stimuli. The recorded temporal-base regions are associated with visual properties including colors, black-and-white information, shapes, and faces. Participants are asked to optimize preprocessing, feature extraction, and classification methods.

Other supported datasets include:

* P300 Speller
* SSVEP
* Motor imagery / stroke rehabilitation
* Locked-in syndrome
* Disorders of consciousness
* ECoG hand pose

This allows the NeuroDecode pipeline to remain dataset-agnostic.

---

# System Architecture

```text
BCI Dataset
     │
     ▼
┌──────────────────────┐
│ Signal Preprocessing │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Feature Extraction   │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Baseline Classifier  │
└──────────────────────┘
     │
     ├───────────────► Prediction
     │
     ▼
┌──────────────────────┐
│ Probability          │
│ Calibration          │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Uncertainty Analysis │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Temporal Stability   │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Selective Prediction │
└──────────────────────┘
     │
     ├── Predict
     │
     └── Abstain
```

---

# Software Structure

The project can be implemented as a reusable Python package:

```text
neurodecode/
├── data.py
├── preprocess.py
├── features.py
├── models.py
├── calibration.py
├── uncertainty.py
├── evaluation.py
├── visualization.py
└── run.py
```

### `data.py`

Responsibilities:

* loading BR41N.IO datasets
* train/test splitting
* trial segmentation
* metadata handling

### `preprocess.py`

Potential preprocessing operations:

* band-pass filtering
* notch filtering
* normalization
* detrending
* artifact rejection
* temporal windowing

### `features.py`

Possible feature families:

* raw signal statistics
* band power
* spectral features
* temporal statistics
* covariance features
* dimensionality-reduced representations

### `models.py`

Initial baseline models could include:

* Linear Discriminant Analysis
* Logistic Regression
* Support Vector Machines
* Random Forest
* shallow neural-network baselines

The goal is not necessarily to build the largest model.

The goal is to understand **when a model's predictions are trustworthy**.

---

# Uncertainty Estimation

Several complementary uncertainty measures can be evaluated.

## Predictive Entropy

Given class probabilities

$$
p(y=k|x)
$$

predictive entropy is:

$$
H(x)=-\sum_k p_k \log p_k
$$

Low entropy indicates a concentrated prediction distribution.

High entropy indicates that the classifier is uncertain between several classes.

---

## Classification Margin

For the two highest predicted class probabilities:

$$
M=p_{(1)}-p_{(2)}
$$

A small margin indicates greater ambiguity.

An uncertainty score can therefore use:

$$
U_{\text{margin}}=1-M
$$

---

## Ensemble Disagreement

Multiple models or perturbed versions of the same model can generate predictions:

$$
p_1(x),p_2(x),...,p_N(x)
$$

Prediction variance can then quantify disagreement.

Large disagreement may indicate that the current sample lies near an unstable decision boundary.

---

## Temporal Instability

BCI predictions occur across time rather than as isolated samples.

Rapidly changing probabilities may therefore indicate unreliable decoding.

For a rolling prediction sequence:

$$
p_t,p_{t-1},...,p_{t-N}
$$

define:

$$
D_t=
\frac{1}{N-1}
\sum_{j=1}^{N-1}
|p_{t-j+1}-p_{t-j}|
$$

Large values indicate unstable predictions.

---

## Combined Uncertainty

A simple combined score could be:

$$
U_t=
\alpha H_t+
(1-\alpha)D_t
$$

where:

* \(H_t\) = predictive entropy
* \(D_t\) = temporal instability
* \(\alpha\) controls their relative weighting

Additional terms could later incorporate:

* ensemble variance
* signal-to-noise measurements
* covariance structure
* effective rank
* model calibration error

---

# Selective Prediction

Traditional classifiers always return a prediction.

NeuroDecode instead allows the system to abstain.

```python
if uncertainty > threshold:
    abstain()
else:
    emit_prediction()
```

Conceptually:

$$
\hat y =
\begin{cases}
f(x), & U(x)<\tau \\
\text{ABSTAIN}, & U(x)\geq\tau
\end{cases}
$$

where:

* \(U(x)\) is decoder uncertainty
* \(\tau\) is an uncertainty threshold

The system therefore trades coverage for reliability.

---

# Main Experiment

The main experiment evaluates whether uncertainty predicts decoder failure.

For every sample or temporal window:

```text
prediction
ground-truth label
confidence
entropy
margin
temporal instability
ensemble disagreement
correct / incorrect
```

The central analysis becomes:

```text
uncertainty at time t
        ↓
probability that prediction is wrong
```

A useful system should assign systematically higher uncertainty to incorrect predictions.

---

# Coverage vs. Accuracy

Selective prediction introduces an important tradeoff:

```text
More predictions
      ↓
greater coverage
      ↓
potentially lower reliability
```

versus:

```text
Fewer predictions
      ↓
lower coverage
      ↓
potentially higher reliability
```

The project will therefore generate a **coverage-vs.-accuracy curve**.

An illustrative result might look like:

```text
Coverage      Accuracy
100%          82%
90%           86%
80%           90%
70%           93%
```

These numbers are only illustrative.

The experiment will determine the actual relationship.

---

# Evaluation Metrics

The project can report several complementary metrics.

### Classification

* accuracy
* balanced accuracy
* F1 score
* confusion matrix
* ROC-AUC where appropriate

### Calibration

* Expected Calibration Error
* Brier score
* reliability diagrams

### Failure Prediction

Treat classifier failure itself as a prediction task:

```text
Will the decoder be wrong?
```

Evaluate uncertainty using:

* ROC-AUC
* precision-recall AUC
* error-detection accuracy

### Selective Prediction

Measure:

* coverage
* selective accuracy
* risk-coverage curve
* area under the risk-coverage curve

---

# Visualization Dashboard

The analysis should produce an easily interpretable report.

Possible outputs:

```text
results/
├── metrics.json
├── confusion_matrix.png
├── calibration_curve.png
├── uncertainty_distribution.png
├── uncertainty_error_curve.png
├── coverage_accuracy.png
├── temporal_predictions.png
└── report.html
```

The demo could play the recorded dataset sequentially and display:

```text
Current Prediction: FACE

Confidence: 0.91
Entropy: 0.24
Temporal Stability: HIGH

Decoder Status:
✓ TRUST PREDICTION
```

during a stable sample.

For an uncertain segment:

```text
Current Prediction: SHAPE

Confidence: 0.51
Entropy: 0.88
Temporal Stability: LOW

Decoder Status:
⚠ ABSTAIN
```

---

# Command-Line Interface

A simple CLI would make NeuroDecode feel like a reusable software tool rather than a one-off notebook.

Example:

```bash
python -m neurodecode.run \
    --dataset ecog_video \
    --model lda \
    --uncertainty entropy \
    --window-ms 500
```

Another experiment could use:

```bash
python -m neurodecode.run \
    --dataset ecog_video \
    --model random_forest \
    --uncertainty ensemble \
    --ensemble-size 32
```

This produces standardized experiment outputs automatically.

---

# Minimum Viable Product

The hackathon MVP only needs five major pieces.

1. Load one BR41N.IO dataset.
2. Train one or more baseline classifiers.
3. Calculate uncertainty for every prediction.
4. Test whether uncertainty predicts classification errors.
5. Generate a coverage-vs.-accuracy analysis.

Everything beyond that is an extension.

---

# Stretch Goals

If time permits, the toolkit could add:

### Ensemble Uncertainty

Train or perturb multiple classifiers and measure prediction disagreement.

### Effective Rank

Analyze whether changes in the covariance structure of neural features correlate with decoding failures.

### Cross-Subject Evaluation

Determine whether uncertainty generalizes between subjects.

### Cross-Session Evaluation

Test whether uncertainty detects distribution shifts between recording sessions.

### Failure Forecasting

Instead of asking:

> Is the current prediction wrong?

ask:

> Does uncertainty now predict decoder failure several windows later?

For horizon \(k\):

$$
U_t
\rightarrow
P(\text{failure}_{t+k})
$$

This would turn uncertainty estimation into a form of **early warning system for BCI degradation**.

---

# Why This Project Is Useful

BCI systems are often evaluated primarily by classification accuracy.

But real-world interfaces require another capability:

> Knowing when not to act.

A reliable neurotechnology system should ideally distinguish between:

```text
"I predict class A."
```

and:

```text
"I predict class A, but this signal appears unreliable."
```

NeuroDecode Lab explores how to make that distinction measurable.

---

# Why It Fits BR41N.IO

The BR41N.IO hackathon explicitly supports offline data-analysis projects where no BCI hardware is required.

The ECoG Video Watching project specifically asks participants to improve preprocessing, feature extraction, and classification approaches using recorded neural data.

NeuroDecode builds directly on that goal while adding uncertainty estimation and selective prediction as an additional reliability layer.

---

# Deliverables

The final hackathon submission would include:

* reusable Python package
* command-line experiment runner
* baseline decoder implementation
* uncertainty estimators
* selective prediction system
* quantitative evaluation
* plots and visualizations
* generated HTML report
* README documentation
* reproducible experiment configuration

---

# Project Pitch

> **NeuroDecode Lab is an uncertainty-aware BCI decoding toolkit that doesn't just predict what a neural signal means—it estimates whether that prediction should be trusted.**

---

# Tagline

## Know When Your BCI Doesn't Know.


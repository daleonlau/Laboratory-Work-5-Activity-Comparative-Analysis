# Laboratory Work 5 — Comparative Analysis of Pre-trained CNN Models
### Euphorbiaceae Family Plant Classification (20 Classes)
**Models Used:** MobileNetV2 · EfficientNetB0 · ResNet50

---

## Table of Contents
- [Project Overview](#project-overview)
- [Training Configuration](#training-configuration)
- [Performance Comparison Table](#performance-comparison-table)
- [Model Architecture Summary](#model-architecture-summary)
- [Evaluation Metrics](#evaluation-metrics)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Guide Questions](#guide-questions)
- [Conclusion](#conclusion)

---

## Project Overview

This laboratory work performs a comparative analysis of three pre-trained CNN models — **MobileNetV2**, **EfficientNetB0**, and **ResNet50** — applied to a custom image dataset of the Euphorbiaceae plant family containing **20 species** with at least **250 images per class**.

All models were trained using **transfer learning** with frozen ImageNet base weights, a `GlobalAveragePooling2D` layer, a `Dense(128, relu)` layer with `Dropout(0.5)`, and a final classification layer.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 pixels |
| Batch Size | 32 |
| Epochs | 10 (MobileNetV2) · 3 (EfficientNetB0, early stopped) · 3 (ResNet50, early stopped) |
| Optimizer | Adam (lr = 0.0001) |
| Loss Function | SparseCategoricalCrossentropy (from_logits=True) |
| Base Model Weights | ImageNet (frozen) |
| Validation Split | 20% |
| Number of Classes | 20 Euphorbiaceae species |
| Images per Class | ≥ 250 |

---

## Performance Comparison Table

| Model | Train Acc | Train Loss | Val Acc | Val Loss | Precision | Recall | F1-Score | ROC AUC |
|---|---|---|---|---|---|---|---|---|
| Teachable Machine (LW2) | ~91.00% | ~0.28 | ~78.00% | ~0.85 | ~0.78 | ~0.78 | ~0.77 | ~0.96 |
| Custom CNN — Model 1 (LW3) | 99.97% | 0.0020 | 85.59% | 1.3288 | 0.73 | 0.71 | 0.71 | 0.924 |
| Enhanced CNN — Model 2 (LW4) | 38.11% | 2.0122 | 44.84% | 1.8479 | 0.48 | 0.45 | 0.44 | 0.784 |
| **MobileNetV2 (LW5)** | **76.42%** | **0.9280** | **83.28%** | **0.7751** | **0.8395** | **0.8328** | **0.8331** | **0.9680** |
| **EfficientNetB0 (LW5)** | **4.68%** | **2.9975** | **4.10%** | **2.9970** | **0.0017** | **0.0410** | **0.0032** | **0.4785** |
| **ResNet50 (LW5)** | **7.60%** | **2.9654** | **8.51%** | **2.9607** | **0.0922** | **0.0761** | **0.0287** | **0.5593** |

> LW3 values from final epoch (Epoch 10/10). LW4 values from final epoch (Epoch 20/20). LW5 values taken directly from the printed comparison table in the Colab notebook output.

---

## Model Architecture Summary

### MobileNetV2
- **Architecture:** Inverted residuals + linear bottlenecks with depthwise separable convolutions
- **Base Parameters:** ~3.4M (frozen)
- **Final Train Accuracy:** 76.42% | **Final Val Accuracy:** 83.28%
- **Strength:** Only model that successfully converged — fastest learning, highest accuracy and AUC in this experiment
- **Weakness:** Validation accuracy plateaued around epoch 6–7, indicating room for improvement via fine-tuning

### EfficientNetB0
- **Architecture:** Compound-scaled MBConv blocks with squeeze-and-excitation modules
- **Base Parameters:** ~5.3M (frozen)
- **Final Train Accuracy:** 4.68% | **Final Val Accuracy:** 4.10%
- **Note:** EfficientNetB0 failed to learn — stopped after 3 epochs with near-random performance. The root cause is a preprocessing incompatibility: placing `Rescaling(1./255)` before the frozen EfficientNetB0 base double-normalized the inputs, as EfficientNet expects raw [0, 255] pixel values and applies its own internal preprocessing. This caused a gradient collapse where all predictions defaulted to a single class.

### ResNet50
- **Architecture:** 50-layer deep residual network with skip connections
- **Base Parameters:** ~25.6M (frozen)
- **Final Train Accuracy:** 7.60% | **Final Val Accuracy:** 8.51%
- **Note:** ResNet50 also failed to converge meaningfully in 3 epochs. With 25.6M frozen parameters and only a small classification head being trained, 3 epochs was insufficient for the head to adapt to 20 plant classes. ResNet50 requires either more epochs or partial unfreezing of the top layers.

---

## Evaluation Metrics

### Classification Report — MobileNetV2

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Acalypha_wilkesiana | 0.83 | 0.98 | 0.90 | 41 |
| Aleurites_moluccana | 0.76 | 0.85 | 0.80 | 48 |
| Euphorbia_heterophylla | 0.97 | 0.82 | 0.89 | 45 |
| Euphorbia_hirta | 0.84 | 0.82 | 0.83 | 44 |
| Euphorbia_marginata | 0.82 | 0.89 | 0.85 | 35 |
| Euphorbia_milii | 0.88 | 0.77 | 0.82 | 47 |
| Euphorbia_neriifolia | 0.83 | 0.83 | 0.83 | 52 |
| Euphorbia_pulcherrima | 0.96 | 0.86 | 0.91 | 63 |
| Euphorbia_tirucalli | 0.85 | 0.92 | 0.88 | 50 |
| Euphorbia_tithymaloides | 0.76 | 0.79 | 0.77 | 52 |
| Hevea_brasiliensis | 0.71 | 0.75 | 0.73 | 53 |
| Hura_crepitans | 0.80 | 0.98 | 0.88 | 46 |
| Jatropha_curcas | 0.73 | 0.89 | 0.80 | 46 |
| Jatropha_integerrima | 0.77 | 0.78 | 0.77 | 59 |
| Jatropha_podagrica | 0.89 | 0.90 | 0.90 | 52 |
| Macarananga_Tanarius | 0.93 | 0.82 | 0.88 | 51 |
| Manihot_esculenta | 0.84 | 0.76 | 0.80 | 54 |
| Ricinus_communis | 0.76 | 0.79 | 0.77 | 47 |
| Triadica_sebifera | 0.89 | 0.81 | 0.85 | 58 |
| Vernicia_fordii | 0.93 | 0.73 | 0.82 | 56 |
| **accuracy** | | | **0.83** | **999** |
| **macro avg** | **0.84** | **0.84** | **0.83** | **999** |
| **weighted avg** | **0.84** | **0.83** | **0.83** | **999** |

### Classification Report — EfficientNetB0

| Metric | Value |
|---|---|
| Overall Accuracy | 4% |
| Macro Avg Precision | 0.00 |
| Macro Avg Recall | 0.05 |
| Macro Avg F1-Score | 0.00 |
| Overall AUC | 0.4785 |

> EfficientNetB0 predicted nearly all 999 validation samples as Acalypha_wilkesiana (one class) due to the preprocessing conflict described above. All other 19 classes received 0.00 precision and 0.00 recall.

### Classification Report — ResNet50

| Metric | Value |
|---|---|
| Overall Accuracy | 8% |
| Macro Avg Precision | 0.09 |
| Macro Avg Recall | 0.09 |
| Macro Avg F1-Score | 0.03 |
| Overall AUC | 0.5593 |

> ResNet50 scattered predictions between a small number of classes (mainly Aleurites_moluccana and Euphorbia_heterophylla columns) with no meaningful learning across all 20 classes.

### ROC AUC Per Class — MobileNetV2

| Class | AUC |
|---|---|
| Acalypha_wilkesiana | 1.00 |
| Aleurites_moluccana | 0.97 |
| Euphorbia_heterophylla | 0.98 |
| Euphorbia_hirta | 0.97 |
| Euphorbia_marginata | 1.00 |
| Euphorbia_milii | 0.94 |
| Euphorbia_neriifolia | 0.97 |
| Euphorbia_pulcherrima | 0.99 |
| Euphorbia_tirucalli | 1.00 |
| Euphorbia_tithymaloides | 0.94 |
| Hevea_brasiliensis | 0.94 |
| Hura_crepitans | 0.99 |
| Jatropha_curcas | 0.98 |
| Jatropha_integerrima | 0.94 |
| Jatropha_podagrica | 0.99 |
| Macarananga_Tanarius | 0.97 |
| Manihot_esculenta | 0.95 |
| Ricinus_communis | 0.93 |
| Triadica_sebifera | 0.96 |
| Vernicia_fordii | 0.96 |
| **Overall AUC** | **0.9680** |

---

## Grad-CAM Explainability

Grad-CAM was applied to all three models using the same Crown of Thorns (*Euphorbia milii*) image showing clusters of pink flowers against green foliage.

### MobileNetV2
- **Heatmap:** Multi-region activation distributed across the upper-left and center of the image
- **Overlay:** Activation spread across several flower heads with partial coverage of surrounding leaves
- **Interpretation:** MobileNetV2 recognized the repeated inflorescence cluster pattern across multiple regions — botanically meaningful, as Crown of Thorns is identified by its clustered cyathia (small flowers surrounded by colorful bracts)

### EfficientNetB0
- **Heatmap:** Single bright activation point in the bottom-right corner; near-zero everywhere else
- **Overlay:** Only the bottom-right corner shows high activation; flower regions receive almost no attention
- **Interpretation:** Completely non-meaningful — consistent with the model's failed training. The single corner activation reflects the collapsed prediction behavior rather than any learned plant feature

### ResNet50
- **Heatmap:** Center-concentrated activation with moderate spread
- **Overlay:** Partial overlap with the central flower cluster, but activation also bleeds into surrounding leaf areas
- **Interpretation:** Slightly more spatially coherent than EfficientNetB0 but still not class-specific. Center bias is a known artifact of ResNet's global average pooling with minimal training epochs

---

## Guide Questions

### A. Model Performance

**1. Which pre-trained model achieved the highest accuracy? Why?**

**MobileNetV2** achieved the highest validation accuracy at **83.28%** with an AUC of **0.9680**. It was the only model among the three that successfully converged within the training window. Its depthwise separable convolution architecture transfers well from ImageNet to fine-grained plant classification — the frozen features generalize effectively to Euphorbiaceae species, and its lightweight design allowed the classification head to learn meaningful decision boundaries in just 10 epochs. Its inverted residual structure also allowed clean gradient flow through the unfrozen top layers even with a frozen base.

**2. Which model had the lowest performance? What could be the reason?**

**EfficientNetB0** had the lowest overall performance with a validation accuracy of **4.10%** and AUC of **0.4785** — essentially random or worse. The cause was a **preprocessing incompatibility**: EfficientNetB0 has its own built-in internal normalization layer and expects raw pixel values in the range [0, 255]. Placing `Rescaling(1./255)` before the frozen EfficientNetB0 base double-normalized the inputs, producing very small values (near 0) that the model's internal batch normalization statistics — calibrated for [0, 255] inputs — could not handle. This caused all predictions to collapse to one class (Acalypha_wilkesiana), as seen in the confusion matrix.

**3. How did loss values compare across models?**

MobileNetV2 was the only model with a meaningful loss trajectory — starting at 3.0449 (epoch 1) and steadily decreasing to 0.9280 by epoch 10, while validation loss dropped from 2.5346 to 0.7751. EfficientNetB0's loss plateaued immediately around ~3.00 for all 3 epochs (final: train 2.9975, val 2.9970) — this is precisely ln(20) ≈ 2.996, the entropy of a uniform distribution over 20 classes, confirming no learning occurred. ResNet50 behaved similarly with a final train loss of 2.9654 and val loss of 2.9607, also anchored at the random-guess baseline.

---

### B. Evaluation Metrics

**4. Why is accuracy not enough to evaluate a model?**

Accuracy only reports the fraction of correct predictions overall and hides per-class failures entirely. In this experiment, accuracy alone is especially misleading: EfficientNetB0 achieved 4% accuracy and ResNet50 achieved 8%, but both models produced completely degenerate confusion matrices — predicting only 1–2 classes for all inputs. Meanwhile the LW3 model achieved 85.59% validation accuracy but had 99.97% training accuracy, revealing severe overfitting that accuracy alone did not expose. For a toxic plant classifier where misidentifying Castor Bean (*Ricinus communis*), Sandbox Tree (*Hura crepitans*), or Pencil Tree (*Euphorbia tirucalli*) as a safe species has serious real-world safety consequences, per-class F1-Score, Precision, Recall, and AUC are essential — they show exactly which dangerous classes are failing, which overall accuracy never reveals.

**5. Which model had the best F1-score? What does it indicate?**

**MobileNetV2** achieved the best weighted F1-Score of **0.8331**. Its per-class F1-scores ranged from 0.73 (Hevea_brasiliensis) to 0.91 (Euphorbia_pulcherrima / Jatropha_podagrica), showing consistently strong performance across all 20 classes without any class completely failing. A high weighted F1-Score means the model balances Precision (few false positives — not labeling other plants as the wrong species) and Recall (few false negatives — not missing actual species) simultaneously across all classes, weighted by class frequency. This is the most important single metric for an imbalanced multi-class plant safety application.

**6. How did Precision and Recall differ across models?**

For **MobileNetV2**, Precision and Recall were closely balanced (weighted avg: Precision 0.84, Recall 0.83). The most notable per-class imbalance was Hura_crepitans: Precision 0.80 but Recall 0.98 — it correctly identified 98% of Sandbox Tree images but sometimes incorrectly labeled other species as Sandbox Tree. Vernicia_fordii showed the opposite: Precision 0.93 but Recall 0.73 — very few false positives but missed 27% of actual Tung Tree images. For **EfficientNetB0** and **ResNet50**, Precision and Recall were essentially meaningless — most classes had 0.00 for both because the models predicted only 1–2 distinct classes across all 999 validation samples.

---

### C. Confusion Matrix Analysis

**7. Which classes were frequently misclassified?**

For **MobileNetV2** (the only meaningful model to analyze):
- **Hevea_brasiliensis** — 5 samples misclassified as Acalypha_wilkesiana, the largest single off-diagonal entry
- **Euphorbia_neriifolia** — 5 samples misclassified as Euphorbia_tirucalli (both are succulent Euphorbias with similar stem morphology)
- **Euphorbia_tithymaloides** — misclassified as Euphorbia_neriifolia and Euphorbia_pulcherrima (3 each)
- **Vernicia_fordii** — scattered misclassifications across multiple classes (lowest recall at 0.73)
- **Jatropha_integerrima** — confused with Jatropha_podagrica (same genus, similar flower appearance)

For **EfficientNetB0**, the full confusion matrix collapsed to one column — all 999 samples were predicted as Acalypha_wilkesiana. For **ResNet50**, predictions clustered in the Aleurites_moluccana and Euphorbia_heterophylla columns with no true diagonal.

**8. What patterns did you observe in the confusion matrix?**

The **MobileNetV2** confusion matrix showed a strong, clean diagonal across all 20 classes — indicating the model learned to distinguish each species effectively. The brightest diagonal values were Euphorbia_pulcherrima (54), Euphorbia_tirucalli (46), Jatropha_podagrica (47), and Triadica_sebifera (47). Off-diagonal entries were sparse, with most misclassifications being 1–3 samples per cell. The **EfficientNetB0** matrix showed a single bright column at Acalypha_wilkesiana with zeros everywhere else on the diagonal — the classic pattern of a model that collapsed to predicting the first class. The **ResNet50** matrix showed two moderately bright columns (Aleurites_moluccana and Euphorbia_heterophylla) with near-zero values everywhere on the diagonal — indicating it learned only the most statistically dominant patterns from those two columns without generalizing.

---

### D. ROC and AUC

**9. Which model had the highest AUC score?**

**MobileNetV2** achieved the highest overall AUC of **0.9680**. At the per-class level, it achieved perfect AUC of **1.00** for three classes: Acalypha_wilkesiana, Euphorbia_marginata, and Euphorbia_tirucalli. The lowest per-class AUC was **0.93** for Ricinus_communis (Castor Bean) — still excellent. EfficientNetB0 scored **0.4785** overall (most classes below 0.5, worse than random) and ResNet50 scored **0.5593** overall (slightly above random but with no class above 0.81).

**10. What does AUC tell us about model performance?**

AUC (Area Under the ROC Curve) measures a model's discriminative ability across all possible classification thresholds, making it completely threshold-independent. An AUC of 1.0 means perfect class separation; 0.5 means the model performs no better than random chance. For the 20-class Euphorbiaceae problem using One-vs-Rest (OvR) strategy, the macro AUC captures how confidently each species is ranked above all others even when a hard-threshold decision at 0.5 might occasionally fail. MobileNetV2's AUC of 0.9680 means it correctly ranks a true species sample above a false one with 96.8% probability. EfficientNetB0's AUC below 0.5 for most classes means it actively anti-ranks — it is more likely to assign higher confidence to the wrong class than the right one, a direct result of the preprocessing collapse.

---

### E. Explainability (Grad-CAM)

**11. What did Grad-CAM reveal about model decision-making?**

The same Crown of Thorns (*Euphorbia milii*) image — pink flower clusters against green foliage — was used for all three models. MobileNetV2's heatmap showed distributed activation across multiple flower clusters, revealing the model recognized the repeated inflorescence pattern that characterizes this species. EfficientNetB0's heatmap showed a single corner activation with no botanical relevance, confirming the training collapse. ResNet50 showed center-concentrated activation that partially overlapped with the main flower cluster but extended into irrelevant background leaf areas.

**12. Did the model focus on relevant image regions?**

Only **MobileNetV2** focused on botanically relevant regions. Its Grad-CAM overlay highlighted multiple pink flower heads distributed across the image frame — exactly the feature a botanist would use to identify Crown of Thorns. The model correctly avoided the pot, background soil, and green leaf edges. **EfficientNetB0** focused on a corner of the image with no plant-related content whatsoever, consistent with its degenerate predictions. **ResNet50** showed partial overlap with the flower region but also activated strongly on surrounding leaves and background, suggesting it captured general color texture rather than species-specific structures.

**13. Which model produced the most meaningful heatmaps?**

**MobileNetV2** produced the most meaningful and botanically interpretable Grad-CAM heatmaps. Its multi-region attention across flower clusters demonstrates that the model correctly identified the repeated inflorescence structure as the discriminating feature for *Euphorbia milii* — consistent with how human botanical experts identify this species. This is further validated by its classification performance: the same model correctly classified 83.28% of validation samples and achieved AUC of 0.9680. ResNet50's heatmap showed partial relevance with noise. EfficientNetB0's heatmap was entirely uninterpretable due to training failure.

---

### F. Model Comparison & Improvement

**14. Which model would you recommend for deployment? Why?**

**MobileNetV2** is the only model from this experiment ready for deployment. It is the only one that learned to classify all 20 Euphorbiaceae species (83.28% val accuracy, 0.8331 F1, 0.9680 AUC). Beyond performance, MobileNetV2's ~3.4M parameter count makes it ideal for mobile deployment as a TensorFlow Lite model — fast inference, small file size, suitable for a camera-based plant identification app. EfficientNetB0 and ResNet50 cannot be deployed in their current state and must be retrained with corrected preprocessing before any comparison is valid.

**15. How can you further improve your best-performing model?**

1. **Fix EfficientNetB0 and ResNet50 preprocessing** — Remove `Rescaling(1./255)` before EfficientNetB0 (use `include_preprocessing=True` or pass raw inputs). Retrain for 10+ epochs. EfficientNetB0 is architecturally stronger and should outperform MobileNetV2 once the pipeline is corrected.
2. **Fine-tune MobileNetV2** — Unfreeze the top 30–40 layers and retrain with lr=1e-5. The current 83.28% val accuracy with a fully frozen base has significant headroom — fine-tuning typically pushes 5–10% higher on custom datasets.
3. **More training epochs** — MobileNetV2's validation accuracy curve was still rising at epoch 10. Training for 20–30 epochs with EarlyStopping would likely push past 87%.
4. **Advanced augmentation** — Add `RandomBrightness`, `RandomContrast`, and `RandomHue` to handle varied field photography conditions that differ from dataset images.
5. **Ensemble** — Once all three models are correctly trained, combine their soft-vote predictions to reduce individual errors, especially on the succulent Euphorbia group.

---

### G. Real-World Application

**16. How can your model be applied in real-world scenarios?**

1. **Mobile plant identification app** — Users photograph a plant in the field and instantly receive the species name, common name, and description (as documented in the LW2 README), helping students, researchers, and the public identify Euphorbiaceae without expert knowledge
2. **Toxicity warning system** — The model triggers safety alerts when dangerous species are detected: Castor Bean (*Ricinus communis*, contains ricin), Sandbox Tree (*Hura crepitans*, explosive fruit + toxic sap), Pencil Tree (*Euphorbia tirucalli*, sap causes blindness), Buddha Belly Plant (*Jatropha podagrica*, all parts toxic)
3. **Agricultural support** — Distinguishing Cassava (*Manihot esculenta*) from toxic weeds and look-alikes in tropical farming regions; identifying Asthma Weed (*Euphorbia hirta*) as a competing weed
4. **Biodiversity monitoring** — Rapid automated plant surveys for ecological research, especially tracking invasive Chinese Tallow Tree (*Triadica sebifera*)
5. **Herbarium digitization** — Automating species classification in botanical museum and university plant collections

**17. What are the risks of deploying an inaccurate model?**

1. **Direct safety risk** — MobileNetV2's 83.28% accuracy means ~1 in 6 images is misclassified. Misidentifying Castor Bean or Sandbox Tree as a safe species could result in poisoning or injury. EfficientNetB0 and ResNet50 in their current state would label every plant as Acalypha_wilkesiana — meaning no safety alerts would ever fire.
2. **Agricultural harm** — False identification of a toxic weed as a crop could lead to contaminated harvests or wrong pesticide decisions
3. **False confidence** — Users trusting the app absolutely, especially in remote areas without botanist access, may make irreversible decisions based on incorrect predictions
4. **Domain shift** — The model was trained on relatively clean, well-framed dataset images. Real field photos with motion blur, partial occlusion, unusual angles, or multiple species in frame will likely degrade performance below the 83.28% benchmark
5. **Silent class failure** — Even with 83.28% overall accuracy, individual classes like Hevea_brasiliensis (F1: 0.73) and Vernicia_fordii (F1: 0.82) perform meaningfully below average — a user encountering these species would face higher misclassification risk without any warning

**18. How can this system be integrated into a mobile/web app?**

1. **Mobile (TensorFlow Lite)** — Convert the MobileNetV2 Keras model using `TFLiteConverter`. Embed the `.tflite` file in Android (TFLite Android SDK) or iOS (Core ML conversion). Camera captures a 224×224 frame, on-device inference returns class label, confidence score, and a red safety warning badge for toxic species.
2. **Web (TensorFlow.js)** — Export using `tensorflowjs_converter` and load in a React or Vue.js frontend. Users upload or capture photos and receive instant browser-side predictions — no server required, works fully offline after initial load.
3. **Backend REST API** — Host the `.keras` model on a cloud server (Google Cloud Run, AWS Lambda, or Render) behind a FastAPI or Flask endpoint. Frontend sends a base64-encoded image; server returns JSON with predicted class, confidence, species description from LW2 README, and a toxicity flag.
4. **Teachable Machine** — The LW2 Teachable Machine model (Google Drive export linked in the LW2 README) can be embedded directly in any webpage using the Teachable Machine JavaScript library for rapid prototyping while the MobileNetV2 model serves as the production backend.

---

## Conclusion

This laboratory work produced critical, honest results. Of the three pre-trained models tested, only **MobileNetV2 successfully converged**, achieving a validation accuracy of **83.28%**, weighted F1-Score of **0.8331**, and AUC of **0.9680**. It significantly outperformed the LW4 enhanced custom CNN (44.84% val accuracy, 0.44 F1) while producing more trustworthy results than the LW3 model — LW3 achieved a higher val accuracy of 85.59% but at the cost of extreme overfitting (99.97% train accuracy), making MobileNetV2's 83.28% a more genuine and generalizable result.

**EfficientNetB0 and ResNet50 both failed to converge** — not due to architectural weakness, but due to a **preprocessing incompatibility** where the `Rescaling(1./255)` layer placed before their frozen bases conflicted with their internal normalization expectations. Both models defaulted to near-random predictions after 3 epochs. This is an important and honest finding: even state-of-the-art architectures fail when input pipelines are misconfigured.

### Key Takeaways
- MobileNetV2 is the only viable model from this experiment and is the sole recommendation for deployment
- EfficientNetB0 and ResNet50 failed due to preprocessing misconfiguration, not architectural inferiority — correcting the input pipeline and retraining is the highest-priority next step
- Transfer learning with MobileNetV2's frozen ImageNet base still significantly outperforms the from-scratch enhanced CNN from LW4, confirming the value of pre-trained features
- LW3's 85.59% val accuracy came with severe overfitting (99.97% train accuracy); MobileNetV2's more balanced train/val gap (76.42% / 83.28%) indicates genuine generalization
- Grad-CAM confirmed MobileNetV2 learned botanically meaningful features (inflorescence cluster patterns for Crown of Thorns), while the other two models' heatmaps reflected their failed training
- The most impactful next step is fixing the preprocessing pipeline for EfficientNetB0 — architecturally, it should outperform MobileNetV2 once correctly configured


### 🔗 Links

| Resource | Link |
|---|---|
| 📓 Colab Notebook | [View Notebook](https://colab.research.google.com/drive/1XO1QfCbD-ebGo-du5DcnVyaHTccJf3g7?usp=sharing) |

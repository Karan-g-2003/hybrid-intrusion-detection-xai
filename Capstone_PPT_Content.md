# Capstone Presentation: Slide Deck Outline
**Project:** Hybrid Intrusion Detection System with Explainable AI (XAI)
*Use this guide to copy-paste directly into your PowerPoint slides. The images mentioned are ready for you in the `outputs/` folder.*

---

## Slide 1: Title Slide
**Title:** Hybrid Network Intrusion Detection System with Explainable AI (XAI)
**Subtitle:** A Temporal Flow Windowing Approach for Modern Cybersecurity
**Presenter:** [Your Name / Team Name]
**Date:** [Presentation Date]

---

## Slide 2: Introduction
**Title:** Introduction to Modern Cyber Threats
- **The Problem:** Modern network threats (DDoS, Botnets, Exploits) are increasingly sophisticated and evasive.
- **The Limitation:** Traditional Intrusion Detection Systems (IDS) evaluate network packets in complete isolation. They act as "black boxes" where security analysts cannot see *why* an alarm was triggered.
- **The Solution:** A Hybrid AI model that looks at the *sequence* of network traffic over time and visually explains its decision-making process to security personnel.

---

## Slide 3: Objectives
**Title:** Project Objectives
1. **Develop** a Hybrid Deep Learning & Machine Learning pipeline capable of capturing sequential attack signatures (using Temporal Windowing).
2. **Solve** the extreme class imbalance problem in cybersecurity datasets (~80% background traffic vs 20% attacks).
3. **Achieve** >95% accuracy on modern benchmark datasets (CIC-IDS-2017).
4. **Implement** an Explainable AI (XAI) engine to map abstract deep features back to original network events (providing transparency to Security Operations Centers).

---

## Slide 4: System Design & Architecture (Overview)
**Title:** High-Level Architecture
- **Stage 1: Data Ingestion & Cleansing:** Merging multi-day network captures and scaling numerical features.
- **Stage 2: Temporal Flow Windowing:** Grouping $N$ consecutive packets to encode the attack timetable.
- **Stage 3: Deep Feature Extraction:** Compressing and analyzing packet sequences using Advanced Neural Networks (Bi-LSTM & Multi-Head Attention).
- **Stage 4: Decision Engine:** Feeding deep network features into an XGBoost classifier for high-precision decision routing.
- **Stage 5: XAI Layer:** Generating SHAP gradients for explainability.

---

## Slide 5: System Architecture (Pipeline Flow)
**Title:** Production Pipeline Flowchart
*(Include standard block diagram or recreate this flow in SmartArt)*
1. **Raw Network Traffic** (CIC-IDS-2017)
2. ⬇️
3. **Temporal Flow Windows** (Size = 10)
4. ⬇️ 
5. **Feature Extractor Module** (Dense $\rightarrow$ Bi-LSTM $\rightarrow$ Attention $\rightarrow$ Global Pooling)
6. ⬇️
7. **XGBoost Classifier Layer**
8. ⬇️
9. **Dual Explainer Output** (SHAP & Gradient Attribution) $\rightarrow$ **Security Alert!**

---

## Slide 6: Implementation Details (Modules)
**Title:** Modules Developed & Integrated
- **Data Pre-Processor Module:** Memory-safe stratified sampling (60k rows/day limit) to ensure model viability on consumer hardware.
- **Temporal Window Generator:** Custom vectorization functions to chunk standard 1D CSV data into 3D Tensors `(Samples, WindowSize, Features)`.
- **Hybrid DL Model Module:** Built using TensorFlow/Keras with modular components (Residual skips + Layer Normalization).
- **Post-Prediction XAI Module:** Built utilizing the `shap` Python library explicitly configured for `TreeExplainer` tree-based evaluation.

---

## Slide 7: Implementation Details (Algorithms - Feature Extraction)
**Title:** Algorithms Implemented (Level 1 Brain)
- **Bi-Directional LSTMs:** Standard LSTMs only read forward in time. We use Bi-LSTMs to read the packet sequence forward and backward, capturing pre- and post-attack signatures.
- **Multi-Head Attention Mechanism (MHA):** Borrowed from LLMs like ChatGPT, MHA instructs our model on *which specific packet* in the 10-packet window is the most suspicious.
- **Focal Loss:** Replaces standard Cross-Entropy. It mathematically forces the AI to ignore easy "Normal Data" and penalize itself heavily if it fails to detect a rare DDoS or DoS attack.

---

## Slide 8: Implementation Details (Algorithms - Decision & XAI)
**Title:** Algorithms Implemented (Level 2 & 3)
- **XGBoost Classifier:** A Gradient Boosted decision tree perfectly suited for tabular networking data. It receives the highly intelligent, compressed "idea" of the Deep Learning model and makes the final categorical classification.
- **SHAP (Shapley Additive Explanations):** Game theoretic approach deployed to calculate exactly how much each mathematical feature contributed to the final detection of the attack.

---

## Slide 9: Code Snippet - The Windowing Logic
**Title:** Code Highlight: Temporal Flow Windowing
*(Take a screenshot of `create_flow_windows` in hybrid_nids_pipeline.py)*
```python
def create_flow_windows(X, y, window_size):
    # Core mathematical reshaping from flat arrays to Time Series
    n_windows = len(X) // window_size
    n_used = n_windows * window_size

    X_windows = X[:n_used].reshape(n_windows, window_size, -1)
    y_grouped = y[:n_used].reshape(n_windows, window_size)

    # Label extraction via Statistical Majority Vote
    y_windows = np.array([np.bincount(row).argmax() for row in y_grouped])
    return X_windows, y_windows
```
*Purpose:* Converts standard tabular AI to sequence-based AI for network mapping.

---

## Slide 10: Results & Analysis (Implementation Check)
**Title:** 100% Implementation Results
- The NIDS pipeline successfully integrates multi-day complex traffic structures into one functional script.
- Validated on over **180,000 traffic flows**.
- Pipeline runs reliably end-to-end, generating models, artifacts, and reports without memory overflows. 
*(Insert image `outputs/class_distribution.png` here to show the dataset balance)*

---

## Slide 11: Results & Analysis (Ablation Verification)
**Title:** Ablation Study Results
- An ablation study trains alternative models on the exact same data to prove our Hybrid is superior.
*(Insert image `outputs/ablation_study.png` here)*
- **XGBoost (Standalone):** 95.1% Accuracy
- **Deep Learning (Standalone):** 97.2% Accuracy
- **Our Hybrid Model:** **97.33% Accuracy** 
- *We successfully pushed macro F1 scores to 90.7% despite severe class imbalances.*

---

## Slide 12: Results & Analysis (Testing Outcomes)
**Title:** Performance Matrix & Classification Reporting
*(Insert image `outputs/confusion_matrix.png` here)*
- **DDoS Detection:** 98% Recall (The model misses almost nothing).
- **Benign Traffic Prediction:** 98% Precision (Zero to few false alarms, ensuring security teams are not burdened).
- Overall Macro Average proves Focal Loss properly handled the minority attack classes.

---

## Slide 13: Results & Analysis (ROC Curve Performance)
**Title:** Area Under the Curve (AUC)
*(Insert image `outputs/roc_curves.png` here)*
- **Explanation:** The ROC curve shows the true positive vs false positive threshold tolerance of our models. 
- The lines hug the top-left quadrant, indicating highly reliable discrimination between attack signatures and safe packets.

---

## Slide 14: Results & Analysis (Explainability Engine)
**Title:** AI Transparency via SHAP
*(Insert image `outputs/shap_deep_features.png` here and/or `gradient_attribution.png`)*
- Shows the top features the model uses to determine an attack.
- We have dissolved the "Black Box." 
- By using gradient extraction, we track down the literal network metrics (e.g., Packet Size, Sequence Flags, Latency bounds) that are triggering the neural network's alarms.

---

## Slide 15: Challenges & Solutions (1)
**Title:** Key Challenges & Resolutions
- **Challenge:** *Extreme Dataset Sizes.* The CIC-IDS-2017 dataset is tens of gigabytes; loading it caused `MemoryError` crashes.
  - **Solution:** Implemented deterministic stratified sampling (capped at 60k rows/day) to guarantee representative traffic distribution without destroying RAM.
- **Challenge:** *Heavy Class Imbalance.* Benign traffic overwhelmed the network gradients, causing standard networks to blindly guess 'Safe' 80% of the time.
  - **Solution:** Replaced standard cross-entropy math with ICCV-2017 Focal Loss algorithms. 

---

## Slide 16: Challenges & Solutions (2)
**Title:** Key Challenges & Resolutions
- **Challenge:** *XGBoost Integration Failure.* You cannot pass 3D Window structures `(None, 10, 78)` natively into XGBoost because it only understands 2D Tables.
  - **Solution:** Orchestrated a hybrid connector: we deployed a `GlobalAveragePooling1D` layer inside the Deep feature extractor space, converting the final complex time-series memory back into an ultra-dense, 1D intelligent feature map perfectly suitable for XGBoost trees.

---

## Slide 17: Remaining Work / Future Scope
**Title:** Future Enhancements
1. **Live Deployment via PCAP parsing:** Integrating tools like Wireshark/PyShark to read raw live packets instead of pre-processed CSVs.
2. **Federated Learning Support:** Allowing multiple universities/companies to collaboratively train the NIDS without sharing private network traffic logs.
3. **Generative Adversarial Network (GAN) stress testing:** Deploying a GAN to automatically generate zero-day attack mutations and feed them recursively back into the pipeline to stress test edge defenses.

---

## Slide 18: Conclusion
**Title:** Thank You / Q&A
- **Summary:** Traditional single-packet IDS limits detection. Our novel Temporal Flow methodology combined with Deep Explainability proves highly formidable and transparent.
- **Live inference test:** (Run `live_demo.ipynb` live right after this slide if the panel allows a demo).
- **Questions?**

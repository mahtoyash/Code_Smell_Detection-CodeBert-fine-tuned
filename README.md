CodeSmellDetector: Hybrid Deep Learning System for Code Quality Analysis
Project Overview
CodeSmellDetector is a specialized machine learning framework designed to automate the detection of technical debt in Python source code. By fine-tuning the microsoft/codebert-base model, this project bridges the gap between static analysis and semantic understanding.

The system categorizes code into five distinct classes: Long Method, Large Parameter List, God Class, Feature Envy, and Clean Code. Unlike traditional tools that rely solely on hard-coded rules, this project uses a hybrid architecture that combines Transformer-based predictions with Abstract Syntax Tree (AST) heuristic overrides for maximum reliability.

Technical Architecture
1. Model Core
The engine uses CodeBERT, a bimodal pre-trained model for programming languages. It has been fine-tuned on a massive dataset of Python snippets, allowing it to understand the structural and functional context of code rather than just keyword matching.

2. Detection Categories (Labels)
Long Method (Label 0): Functions exceeding 20 lines or a Cyclomatic Complexity (CC) score of 10.

Large Parameter List (Label 1): Functions defined with more than 5 parameters, hindering maintainability.

God Class (Label 2): Massive classes that handle too many responsibilities (defined by 7+ methods and 10+ attributes).

Feature Envy (Label 3): Methods that demonstrate a higher coupling with external objects than their own class members.

Clean Code (Label 4): Standard compliant code that serves as a negative control for training.

3. Data Engineering Pipeline
Production Mining: The system includes a dedicated miner that clones and scans enterprise-grade repositories (including Django, Pandas, and Odoo) to extract real-world instances of code smells.

Synthetic Generation: To solve class imbalance issues, a custom synthetic engine generates over 60,000 samples including "Hard Negatives"—code that looks like a smell (e.g., a Dataclass with many attributes) but is architecturally sound.

Stratified Balancing: The dataset is processed through a stratified split to ensure equal representation during the fine-tuning process.

Hybrid Inference Engine
A key differentiator of this project is the Heuristic Fallback Layer.
The system does not rely blindly on the neural network. For high-impact labels like "God Class," the inference server runs a secondary check using Python's ast module. If the model predicts a God Class but the structural metrics do not meet a secondary safety threshold, the system can flag the result for manual review or apply a heuristic correction.

System Performance
Model Optimization: Implements Gradient Accumulation and FP16 Mixed Precision training for efficient VRAM usage.

Evaluation Metrics: Beyond standard accuracy, the project evaluates performance using Macro F1-Score to ensure high precision across imbalanced real-world data.

Explainability: Includes a confusion matrix generation utility to visualize inter-class misclassifications (e.g., distinguishing between a Long Method and Feature Envy).

Installation and Deployment
Environment Setup
The project requires Python 3.10 and the following libraries:

PyTorch

Transformers (HuggingFace)

Datasets

Radon (for Cyclomatic Complexity)

Flask (for API deployment)

API Usage
The system deploys a Flask REST API on port 5000.

Endpoint: /predict

Method: POST

Payload: {"code": "string"}

Response: Returns the predicted smell, confidence score, and raw model logits.

Future Roadmap
Integration as a GitHub Action for automated PR reviews.

Expansion to support Java and C++ using multi-lingual CodeBERT.

Development of a VS Code extension for real-time IDE feedback.

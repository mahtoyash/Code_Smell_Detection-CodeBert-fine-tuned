CodeSmellDetector: Hybrid Machine Learning for Technical Debt Detection

1. Project Description
CodeSmellDetector is a specialized software quality tool that uses Deep Learning to identify architectural and structural "smells" in Python code. While traditional static analyzers (like Pylint or Flake8) use hard-coded rules, this project utilizes a fine-tuned CodeBERT transformer model to understand the semantic intent and complexity of code.

The project addresses the challenge of identifying technical debt that is often missed by standard linting tools, such as Feature Envy or God Classes, which require a deeper understanding of class-method relationships.

2. Technical Features
Model Core: Fine-tuned microsoft/codebert-base on a dataset of over 60,000 code snippets.

Hybrid Architecture: Combines probabilistic Deep Learning predictions with a deterministic Heuristic Fallback layer using Python's Abstract Syntax Tree (AST).

Automated Data Mining: Includes scripts to clone and analyze top-tier open-source repositories (Django, Pandas, Scikit-learn) for real-world code smell extraction.

Synthetic Data Engine: Developed a custom generator to create "Hard Negatives" (code that appears to be a smell but is architecturally correct) to improve model precision.

3. Classification Categories
The system is trained to detect five specific categories:

Long Method: Functions that have excessive lines of code or high cyclomatic complexity.

Large Parameter List: Functions that accept too many arguments, making them difficult to test and maintain.

God Class: Large classes that have centralized too much intelligence and responsibility (High Method/Attribute count).

Feature Envy: Methods that are more coupled to external objects than to the class they belong to.

Clean Code: High-quality code used as a control group to minimize false positives.

4. System Implementation
Backend: PyTorch and HuggingFace Transformers for model training and management.

Static Analysis: Integrated radon for cyclomatic complexity and ast for structural parsing.

Deployment: A Flask-based REST API that accepts raw code strings and returns a detailed JSON analysis including confidence scores.

Optimization: Utilized Mixed Precision (FP16) and Gradient Accumulation to train large transformer models on consumer-grade hardware.

5. Evaluation Metrics
The project focuses on Macro F1-Score rather than simple Accuracy to ensure the model performs reliably across all categories, especially for rarer smells like God Classes. It includes a "Real-Code Only" evaluation module to validate model performance on production-grade snippets versus synthetic examples.

6. Setup and Usage
The system requires Python 3.10+ and the following dependencies:

torch

transformers

datasets

flask

radon

To use the system, run the inference server and send a POST request to the /predict endpoint with the code snippet in the request body.

7. Future Scope
Development of a CLI tool for local repository auditing.

Support for multi-file analysis to detect Cross-File Feature Envy.

Integration with CI/CD pipelines to block commits that introduce significant technical debt.

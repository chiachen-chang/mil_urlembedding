# URL Embedding Using MIL Algorithm

**Paper:Bag-of-Characters: A Multiple Instance Learning Framework for URL Embedding in Web Security**

## Project Overview
This project implements a novel approach for embedding URLs using a Multi-Instance Learning (MIL) approach. The goal is to enhance the detection of malicious web activities by transforming URLs into a structured, vectorized format that captures both semantic and structural nuances.

## Modules
The project is divided into three main modules:
1. **`data_preprocessing.py`** - Handles the loading and initial processing of URL data from CSV files.
2. **`feature_extraction.py`** - Manages the transformation of URLs to vector representations using position encoding and normalizes these vectors using a MIL-based strategy.
3. **`main.py`** - Orchestrates the training process, applies KMeans clustering, computes miVLAD vectors, and saves the results.


## Getting Started
To get started with this project, clone the repository and install the required dependencies:
```bash

git clone https://github.com/chiachen-chang/mil_urlembedding
cd your-repository-directory
pip install -r requirements.txt
```

## Usage
Run the `main.py` to start the process:
```bash
python main.py
```

## Contributing
We welcome contributions from the community, whether they are feature requests, improvements, or bug fixes. Please fork the repository and submit your pull requests for review.

## Discussion and Learning
We encourage everyone to participate in discussions and learning around this project. If you have questions, suggestions, or insights, please feel free to open an issue for discussion

Let's collaborate to make URL embedding even more effective and secure!


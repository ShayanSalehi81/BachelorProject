Here's a comprehensive README for your repository:

---

# AYA Prompt-Based Classification and Evaluation

This repository provides a framework for prompt-based classification using pre-trained language models, with a focus on Persian text classification tasks. It includes scripts and notebooks for generating prompts, fine-tuning prompts for classification, evaluating results, and analyzing model performance metrics such as F1 score, precision, and recall. The repository also supports K-shot learning to enhance model adaptability by incorporating relevant examples.

## Project Structure

- **Codes**: Contains the core code and notebooks for model training, prompt generation, and evaluation.
  - `AYA-Colab.ipynb`: Main notebook for training and fine-tuning prompts with AYA models on Colab.
  - `Classification_report.ipynb`: Generates classification metrics, including F1 score, precision, and recall for different prompt setups.
  - `Creating_dataset.ipynb`: Data preparation and dataset creation for prompt-based learning.
  - `f1-calculation.py`: Python script to calculate and visualize F1 scores.
  - `news-aya-symbol-tuning.ipynb`: Notebook for symbol-based tuning with AYA models for text classification.
  - `news-aya-system-user-prompt.ipynb`: Script for generating system and user prompts using a pre-trained language model.
  - `Symbol_tuning_aya.ipynb`: Symbol tuning notebook for optimizing prompt effectiveness.

- **Datasets**: Contains datasets used for training and evaluation.
- **Prompts**: Contains prompt templates used for various classification tasks.
- **Slides**: Documentation and presentation files explaining in-context learning, prompt design, K-shot learning, and symbol tuning.
  - `In-Context Learning.pptx` & `In-Context Learning.pdf`: Details on using in-context learning for model tuning.
  - `System-User Prompt Design.pptx` & `System-User Prompt Design.pdf`: Guide for designing system and user prompts.
  - `Symbol Tuning.pptx` & `Symbol Tuning.pdf`: Instructions on using symbol tuning to improve prompt performance.

## Key Features

- **Prompt-Based Classification**: Framework to classify text using prompts with a language model. The system allows dynamic generation of prompts, integrating user-defined inputs and system prompts for flexible text classification.
- **K-Shot Learning**: Supports K-shot learning where the model is provided with K relevant examples to improve performance on specific tasks.
- **Evaluation Metrics**: Provides tools for comprehensive evaluation, including accuracy, F1 score, precision, and recall. Results are saved and can be visualized through confusion matrices and classification reports.
- **Symbol Tuning**: Techniques to adjust and refine prompts by using symbols and other prompt-based modifications, enhancing model responsiveness to specific queries.
- **In-Context Learning**: Documentation and support for in-context learning to improve prompt-based model adaptability with examples in the prompt context.

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/aya-prompt-classification.git
   cd aya-prompt-classification
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate with Hugging Face (if necessary) and install additional libraries:

   ```bash
   huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
   ```

4. Run any of the notebooks or Python scripts in the `Codes` directory to perform tasks such as dataset creation, prompt tuning, or evaluation.

## Usage

### Generating Prompts and Running Classification

- **news-aya-system-user-prompt.ipynb**: This notebook provides an end-to-end pipeline for generating system and user prompts and performing classification on news datasets. The `Generator` class loads a pre-trained language model, formats prompts, and generates predictions. The script supports 4-bit quantization for efficient memory usage and leverages user-provided prompts to classify Persian news data as "important" or "not important."

### Evaluation and Metrics

- **Classification_report.ipynb**: Evaluates model performance with metrics such as accuracy, precision, recall, and F1 score. It includes K-fold cross-validation and produces detailed classification reports.
- **f1-calculation.py**: Calculates and visualizes F1 scores for classification results, with category-wise breakdowns. Confusion matrices and summary tables can be generated to understand model performance across categories.

### K-Shot Learning

- The prompt generation pipeline supports K-shot learning, where K most similar examples are retrieved from the training set using TF-IDF similarity. This enhances prompt-based classification by providing the model with contextually relevant examples.

### Symbol Tuning

- Notebooks like `news-aya-symbol-tuning.ipynb` and `Symbol_tuning_aya.ipynb` are designed to fine-tune prompt symbols, which can improve model interpretability and response consistency. Symbol tuning introduces minor adjustments to the prompts, enhancing the model's comprehension of nuanced queries.

## Example Workflow

1. **Data Preparation**: Use `Creating_dataset.ipynb` to preprocess and format your dataset.
2. **Prompt Generation**: Load `news-aya-system-user-prompt.ipynb` to define system and user prompts, and run classification on the dataset.
3. **Evaluation**: Use `Classification_report.ipynb` to calculate metrics like accuracy and F1 score and `f1-calculation.py` to visualize performance.
4. **Symbol Tuning**: Run `news-aya-symbol-tuning.ipynb` to refine prompt design with symbol tuning.

## Future Enhancements

- **Prompt Optimization**: Further refine prompt generation methods to support more complex classification tasks.
- **Fine-Tuning**: Incorporate model fine-tuning on custom datasets to improve model adaptability.
- **Extended K-Shot Learning**: Experiment with variable K values to optimize in-context learning.
- **Symbol Tuning Enhancements**: Extend symbol tuning techniques to handle a broader range of tasks and user contexts.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to enhance this project.

--- 

This README provides an overview of the functionality, setup, and usage of your prompt-based classification framework, making it easier for users to understand and utilize the project effectively.
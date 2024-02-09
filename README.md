# USMLE First-Aid Chatbot

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

This project, built using [LangChain](https://langchain.org/) with [Qdrant](https://qdrant.io/) as the vector database, incorporates the innovative concept of Response-Action-Goal (RAG) for chatbot functionality. Leveraging the power of [Hugging Face](https://huggingface.co/) embeddings and [MistralAI/Mixtral-8x7B-Instruct-v0.1](https://github.com/Mistral-8x7B/Mistral-8x7B-Instruct-v0.1) as its large language model (LLM), the project is designed to provide intelligent responses.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- [Docker](https://docs.docker.com/get-docker/): Make sure Docker is installed on your machine.
- [Conda](https://docs.conda.io/en/latest/miniconda.html): Preferred for creating a virtual environment. Install Miniconda or Anaconda if you don't have Conda installed.

## Installation

1. **Clone the project repository:**

   ```bash
   git clone https://github.com/Sujen-Shrestha/langchain.git
   cd your-project
    ```bash

2. **Create a Conda Virtual Environment:**

    # Create a virtual environment
    ```bash
    conda create --name your-environment-name python=3.8.10
    ```bash
    
    # Activate the virtual environment

    ```bash
    conda activate your-environment-name
    ```bash

    # Install required packages
    
    ```bash
    pip install -r requirements.txt
    ```bash

3. **Set Up Qdrant with Docker:**
    # Pull Qdrant Docker image
    ```bash
    docker pull qdrant/qdrant
    ```bash

    # Run Qdrant container
    ```bash
    docker run -p 6333:6333 \
        -v $(pwd)/path/to/data:/qdrant/storage \
        qdrant/qdrant
    ```bash

    Note: The above Docker commands are written for Unix-like systems. Adjust the volume mounting syntax accordingly if you're using Windows.

    #Make sure Qdrant is running before proceeding to the next steps.
    # Confirm Qdrant is running

    ```bash
    docker ps
    ```bash

## Usage

1. **Create a new terminal:**
   
   Open a new terminal window. Note: You may need to activate the virtual environment for this terminal as well.

2. **Change directory to go to the chatbot folder:**
   
   ```bash
   cd chatbot
   ```bash

3. **Run document_ingestion.py:**

    ```bash
    python document_ingestion.py
    ```bash

4. **Run main.py:**

    ```bash
    python main.py
    ```bash
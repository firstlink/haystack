# Haystack2.0

Introduction to Haystack2.0 library. Build customizable LLM pipelines with AI Tools

## Description  

This repository contains real world examples using Haystack2.0 library

## Dependencies 

* haystack-ai = "*2.0.0"
* datasets = "*"
* sentence-transformers = "*"
* wikipedia = "*"

## Getting Started

### Installation

To install the required dependencies, you can use pipenv.  
1. Make sure you have `Pipenv` installed on your system.  
2. Navigate to the root directory of this project and run: `pipenv install`

### OpenAI Configuration Steps

To configure OpenAI's GPT, follow these steps:

1. Sign Up for OpenAI: If you haven't already, sign up for an account on the OpenAI platform.  
2. Get API Key: Once you have an account, navigate to the API section and generate an API key.  
3. Configure Environment Variable: In python files replace environment variable value "<open-ai-key>" to "API-Key" from OpenAI

### Run program  

1. From your terminal navigate to root project  
2. pipenv run python <file-name>.py
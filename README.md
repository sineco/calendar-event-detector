# Calendar Event Detector

## Overview
This project processes real-time chat streams to detect calendar events using a fine-tuned BERT model hosted on Hugging Face. It is wrapped in a Docker container for easy deployment and evaluation.

## Features
- **Real-Time Processing**: Handles WebSocket streams and detects entities like `DATE`, `TIME`, and `MEETING PLATFORM`.
- **Fine-Tuned BERT Model**: Uses `bert-finetuned-ner` from Hugging Face for entity recognition.

## Model Information
The fine-tuned BERT model (`bert-fine-tuned-ner`) is hosted on Hugging Face and will be automatically downloaded when the application runs. This approach was taken to avoid uploading the full model to GitHub due to its large size. However, the script used to fine-tune the model is available in the repository at `fine_tuning_scripts.py`.

For more details about the model, visit the [Hugging Face Model](https://huggingface.co/raraujo/bert-finetuned-ner).


## File Descriptions
- **main.py**: WebSocket client to receive and process real-time messages.
- **event_detector.py**: Contains the logic for detecting calendar events using the fine-tuned model.
- **predict_ner.py**: Script for testing the model on individual samples.
- **requirements.txt**: Lists all Python dependencies.
- **Dockerfile**: Instructions to create a Docker image for the project.


## Setup and Deployment

## Prerequisites
1. **Docker**: Install Docker from [here](https://docs.docker.com/get-docker/).
2. **Git**: Ensure Git is installed. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### Requirements
- Docker (latest version)

### Building and running the Docker Image
Clone this repository and navigate to the project folder:
```bash
git clone https://github.com/sineco/calendar-event-detector.git
cd calendar-event-detector
```

Now build the Docker image using the following command:
```bash
docker build -t calendar-event-detector .
```

Run the Docker image
```bash
docker run -it --rm calendar-event-detector
```
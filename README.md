# Calendar Event Detector

## Overview
This project processes real-time chat streams to detect calendar events using a fine-tuned BERT model hosted on Hugging Face. It is wrapped in a Docker container for easy deployment and evaluation.

## Features
- **Real-Time Processing**: Handles WebSocket streams and detects entities like `DATE`, `TIME`, and `MEETING PLATFORM`.
- **Fine-Tuned BERT Model**: Uses `bert-finetuned-ner` from Hugging Face for entity recognition.

## Setup and Deployment

## Prerequisites
1. **Docker**: Install Docker from [here](https://docs.docker.com/get-docker/).
2. **Git**: Ensure Git is installed. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### Requirements
- Docker (latest version)

### Building the Docker Image
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
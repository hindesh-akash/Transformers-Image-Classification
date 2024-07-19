# Image Classification Service

This repository provides an image classification service using a pre-trained machine learning model. The service is deployed using Docker and exposes an API for processing images.

## Table of Contents

- [Image Classification Service](#image-classification-service)
  - [Table of Contents](#table-of-contents)
  - [Deployment Instructions](#deployment-instructions)
    - [Prerequisites](#prerequisites)
    - [1. Build the Docker Image](#1-build-the-docker-image)
    - [2. Run the Docker Container](#2-run-the-docker-container)
  - [API Usage](#api-usage)
    - [Endpoint](#endpoint)
    - [Example Request](#example-request)
  - [Testing](#testing)
  - [Contributing](#contributing)
  - [License](#license)

## Deployment Instructions

### Prerequisites

Ensure you have Docker installed on your system. If not, you can download and install it from the [official Docker website](https://www.docker.com/get-started).

### 1. Build the Docker Image

Open a terminal and navigate to the directory containing the `Dockerfile`. Run the following command to build the Docker image:

```sh
docker build -t image-classification-service .

# Mistral in Pytorch

## Overview
This projects implements [Mistral](https://arxiv.org/pdf/2310.06825.pdf) transformer architecture for self-supervised prediction, which is at the core of LLMs. It aims to provide a simple and efficient implementation of popular Mistral model which is based on the original [transformer architecture](https://arxiv.org/abs/1706.03762) which is highly flexible and powerful, but implements few upgrades such as: [rotary embeddings](https://arxiv.org/pdf/2104.09864.pdf), [grouped query attention for a tradeoff between MHA and MQA](https://arxiv.org/abs/2305.13245v3), [SwiGLU](https://arxiv.org/abs/2002.05202v1), [RMS Norm](https://arxiv.org/abs/1910.07467) [KV Caching](https://arxiv.org/pdf/2211.05102.pdf) and [sliding window attention](https://arxiv.org/pdf/2004.05150v2.pdf).

## Architecture

![Mistral](https://miro.medium.com/v2/resize:fit:1400/1*cG4isCiXyMQ9sSlH8wj_NQ.png)

 The Mistral architecture consists of the Transformer Decoder architecture, coupled with few upgrades such as :
 * Rotary Encodings
 * SwiGLU
 * Prefill and Chunking
 * Rolling Buffer Cache
 * Grouped Query Attention
 * Sliding Window Attention


**Decoder**: The decoder takes in the output of the encoder and generates the final output sequence. It also consists of a stack of decoder layers. Each decoder layer has a grouped query multi-head self-attention mechanism, feed-forward neural network.
It benefits from RoPe encodings, KV caching and everything mentioned above.

 **Grouped Query Attention**: The grouped query attention mechanism is a modification to the traditional attention mechanism in the transformer architecture. It allows the model to attend to different groups of queries within the input sequence, enabling a tradeoff between multi-head attention and multi-query attention. This helps improve the model's ability to capture complex dependencies and relationships within the data.


For more details on the transformer architecture, refer to the original paper: [Mistral](https://arxiv.org/pdf/2310.06825.pdf).



## Setup

To get started with Transformer Plain, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/paulilioaica/Mistral
    cd Mistral-Pytorch/

    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

## License

This project is licensed under the MIT License. 

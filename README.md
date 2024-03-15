# ai-agents

This repository is a Rust library designed for building and managing generative AI agents, leveraging the capabilities of large language models (LLMs), such as ChatGPT. The aim of this project is to provide a robust and scalable framework that is adaptable to a wide range of scenarios. (`ai-agents` is at **a very early stage** of development.)

## Key Features

- **Modular Architecture**: Enhances the creation of complex AI behaviors by chaining together smaller, interoperable components. This is reusability and flexibility across different projects.
- **Agent and Module Traits**: a flexible interface for building custom agents and modules, allowing for the execution of sophisticated AI tasks with ease.
- **Asynchronous Support**

## Crates

- `ai-agent-macro`
- `sllm-rs`: A crate dedicated to interfacing with Large Language Models (LLMs), including utilities for sending requests and processing responses.


## Examples

The following examples are simulations of limited situations, demonstrating the application of `ai-agents` to specific scenarios:


### Run Examples

To run the examples, you need to set an environment variable `OPEN_API_KEY` with your API key. This can be done by creating a `.env` file in the root of the project.

```
OPEN_API_KEY=your_api_key_here
```

- **Find Treasure**: A game simulation where the player's goal is to find treasure in a dynamically generated scenario by interacting with NPCs.
  ```
  cargo run -p find-treasure
  ```

- **Ecommerce Chat Assistant**: A limited simulation agent that, based on customer inputs (such as name and order ID), explains the current state of an order.
  ```
  cargo run -p ecommerce-assistant
  ```

# ecommerce-chatbot

A simple e-commerce agent (chatbot) designed to handle customer support queries. This tool greatly improved data visibility and decision-making efficiency.

## Description
This chatbot is built using Python and utilizes machine learning techniques to understand and respond to user inputs.

## Built With

- [Python](https://www.python.org/)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://www.langchain.com/langgraph)
- [Google API](https://aistudio.google.com/app/)
- [OpenAI API](https://openai.com/index/openai-api/)

## Installation

1. **Clone the repository:**

    ```bash
    https://github.com/savannahtech/insait-llm-charles
    cd ecommerce-chatbot
    ```

2. **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    For windows users, replace the `source venv/bin/activate` with `venv\Scripts\activate` to activate the environment.

3. **Install the necessary dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the project root and add the necessary configuration details (e.g., API keys, database URLs). An example of what is requires is shown in the `dotenv` file

    1. **Get a Google or OpenAI API Key:**
        - Sign up at [Google AI Studio](https://aistudio.google.com/app/apikey) or [OpenAI API](https://openai.com/index/openai-api/).
        - Click **Create API key**.
        - Copy your API key.

    2. **Add the following to your `.env` file:**

    ```plaintext
    GOOGLE_API_KEY=your-google-api-key
    OPENAI_API_KEY=your-openai-api-key
    ```

## Usage

Here are some examples of how to use the chatbot:

1. **Running the chatbot locally:**

    Interact with the chatbot iteratively by running:

    ```bash
    python main.py
    ```

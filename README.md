# conversation-chatbot
## Python Chatbot with Conversation Memory

A simple Python-based chatbot that uses OpenAI's GPT model (e.g., GPT-5) to provide sarcastic, humorous replies to user input via the command line. This version **remembers the conversation** across turns, building context as you chat.

### Features

- Conversational interface in your terminal.
- Sarcastic, humorous personality powered by OpenAI's GPT.
- **Conversation memory**: each exchange is retained for context.
- Handles errors gracefully.

### Requirements

- Python 3.7+
- An OpenAI API key ([get one here](https://platform.openai.com/account/api-keys))
- The `openai` Python library

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/qhaydar/conversation-chatbot.git
    cd conversation-chatbot
    ```

2. **Install dependencies:**

    ```bash
    pip install openai
    ```

3. **Set your OpenAI API key as an environment variable:**

    On Unix/Mac:
    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

    On Windows:
    ```cmd
    set OPENAI_API_KEY=your_openai_api_key
    ```

### Usage

Run the chatbot in your terminal:

```bash
python main.py
```

You'll see a prompt:

```
You: 
```

Type your message and hit Enter. To exit, type `bye`.

### Example Conversation

```
You: hello
Chatbot: Oh joy, another human. What *extraordinary* thing can I do for you today?
You: How's the weather?
Chatbot: It's just as you'd expect—outside, mainly. Would you like me to check your window for you?
You: tell me a joke
Chatbot: Why did the AI cross the road? Because it was programmed to get to the other side of the data.
You: bye
Chatbot: Bye!
```

### Notes

- The chatbot replies will reflect a sarcastic, humorous tone thanks to the system prompt configuration.
- Conversation memory means the model sees the full chat history, enabling more coherent and context‑aware responses.
- If you receive errors, ensure your OpenAI API key is set correctly and you have an active internet connection.
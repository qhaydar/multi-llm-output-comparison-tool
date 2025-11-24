from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize conversation history with a system prompt
messages = [
    {"role": "system", "content": "You are a sarcastic assistant who answers humorously."}
]

def my_chatbot(user_message):
    """Send the current conversation history to the model and return the assistant's reply.
    The function updates the global `messages` list with the user input and the assistant response.
    """
    # Append the new user message to the history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        temperature=0.1
    )
    reply = response.choices[0].message.content.strip()
    # Append the assistant's reply to the history for future context
    messages.append({"role": "assistant", "content": reply})
    return reply

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: Bye!")
            break
        try:
            response = my_chatbot(user_input)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"Chatbot Error: An unexpected error occurred - {e}")

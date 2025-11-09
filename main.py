from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def my_chatbot(user_message):
    response = client.chat.completions.create(model="gpt-5",
    messages=[{
        "role": "system", "content": "You are a sarcastic assistant who answers humorously."
        "content": user_message
    }],
    temperature=0.5)

    return response.choices[0].message.content.strip()

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
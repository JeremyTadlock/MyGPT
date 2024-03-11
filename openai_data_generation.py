import openai
#import cchardet as chardet
import time

data = ""

your_api_key = "[keep secret dont upload to github please]"
openai.api_key = your_api_key

messages = [
    {"role": "system", "content": "You are a helpful assistant."},

]
new_msg =  "Please create a unique text for the purpose of being added to a dataset to train a simple bigram gpt model using a BPE tokenizer. When creating this text, please only use the oxford 3000 most common english words. please answer in 3000-4000 characters"
followup_msg = "Thank you. Keeping my first question in mind, can you generate another unique example for me? try to make it unique and different than the others, as i will be combine all of your responses together."
messages.append({"role": "user", "content": new_msg})
chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
reply = chat.choices[0].message.content
data = data + " " + reply
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": followup_msg})
for x in range(2000):
    try:

        print(x)
        print(reply)
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        #print(len(chat.choices))
        reply = chat.choices[0].message.content
        if x % 100 == 0 or x == 5000:
            print(reply)
        #print(reply)
        #messages.append({"role": "assistant", "content": reply})
        #messages.append({"role": "user", "content": followup_msg})
        data = data + " " + reply
        #print(data)
    except openai.error.RateLimitError as e:
        # wait for 30 minutes before making next request
        print("Rate limit exceeded. Waiting for 10 minutes...")
        time.sleep(10 * 60)
    except Exception as e:
        # handle other exceptions here, also wait 30 minutes just in case.
        print("Error:", e)
        time.sleep(10 * 60)

with open("openai_generated_text_3000_1.txt", "w", encoding="utf-8") as f:
    f.write(data)
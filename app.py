import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer
import matplotlib.pyplot as plt


model_path = 'berta'
model = tf.keras.models.load_model(model_path)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def plot_pie_chart(class_probabilities, emotion_labels):
    class_probabilities = class_probabilities.numpy().flatten()

    fig, ax = plt.subplots()
    ax.pie(class_probabilities, labels=emotion_labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  

    plt.title("Emotion Class Probabilities")
    # plt.show()
    st.pyplot(fig)

emotion_labels = sorted(['joy', 'sadness', 'nervousness', 'anger', 'disgust', 'optimism', 'grief', 'neutral'])

def predict_emotion(text):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True
    )

    input_ids = tf.convert_to_tensor([tokens['input_ids']])
    attention_mask = tf.convert_to_tensor([tokens['attention_mask']])
    token_type_ids = tf.convert_to_tensor([tokens['token_type_ids']])

    predictions = model.predict({'input_ids': input_ids,
                                 'attention_mask': attention_mask,
                                 'token_type_ids': token_type_ids})

    # predicted_class = tf.argmax(predictions.logits, axis=1).numpy()[0]
    predicted_class_probs = tf.nn.softmax(predictions['logits'], axis=1)

    return predicted_class_probs

# Streamlit UI
st.title("Emotion Detection App")

user_input = st.text_area("Enter text:")
if user_input:
    # predicted_emotion = predict_emotion(user_input)
    st.write(f"Predicted Emotion")
    class_probabilities = predict_emotion(user_input)
    plot_pie_chart(class_probabilities, emotion_labels)



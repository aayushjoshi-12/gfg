import string
from transformers import BertTokenizer
import tensorflow as tf

def drop_labels(labels, reduced_labels, df):
    for label in labels:
        if label not in reduced_labels:
            df = df.drop(label, axis=1)

def accumulate(row):
    return list(row.values)

# example use:
# df['label'] = df.iloc[:, 2:].apply(accumulate, axis=1)

def remove_numbers(text):
    return ''.join(char for char in text if not char.isdigit())

def remove_special_characters(text):
    return ''.join(char for char in text if char not in string.punctuation)

# example use:
# for i in range(len(df['Text'])):
#     df['Text'][i] = remove_special_characters(df['Text'][i])
#     df['Text'][i] = remove_numbers(df['Text'][i])
#     df['Text'][i] = df['Text'][i].lower()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # hyperparameter to change 
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True
    )

def convert_to_tf_dataset(texts, labels, batch_size=32):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for text in texts:
        tokens = tokenize_text(text)
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
        token_type_ids.append(tokens['token_type_ids'])

    input_ids = tf.convert_to_tensor(input_ids)
    attention_masks = tf.convert_to_tensor(attention_masks)
    token_type_ids = tf.convert_to_tensor(token_type_ids)
    labels = tf.convert_to_tensor(labels)

    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks, 'token_type_ids': token_type_ids}, labels))

    return dataset.batch(batch_size)

# example use:
# texts = list(df['Text'])
# labels = list(df['label'])
# batch_size = 8 # another hyperparameter we will fine tune later
# train_dataset = convert_to_tf_dataset(texts, labels, batch_size=batch_size)
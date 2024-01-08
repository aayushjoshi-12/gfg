import tensorflow as tf
from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)


optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5) # hyperparameter to change 
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# model.fit(train_dataset, epochs=10)

model.save('berta')
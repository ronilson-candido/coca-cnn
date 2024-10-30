import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')


print(train_df.head())
print(test_df.head())
print(sample_submission.head())


train_df['image_path'] = './data/images/' + train_df['id']
test_df['image_path'] = './data/images/' + test_df['id']


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col=['CONTENT_HIGH', 'CONTENT_LOW', 'COVER_NONE', 'BOTTLE_SMASHED', 'LABEL_WHITE', 'LABEL_MISPLACED', 'LABEL_NONE', 'BOTTLE_NONE'],
    target_size=(128, 128),
    batch_size=32,
    class_mode='raw',
    subset='training'
)


validation_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col=['CONTENT_HIGH', 'CONTENT_LOW', 'COVER_NONE', 'BOTTLE_SMASHED', 'LABEL_WHITE', 'LABEL_MISPLACED', 'LABEL_NONE', 'BOTTLE_NONE'],
    target_size=(128, 128),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='sigmoid')  
])


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(train_generator, epochs=10, validation_data=validation_generator)


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

predictions = model.predict(test_generator)


submission = sample_submission.copy()
submission[['CONTENT_HIGH', 'CONTENT_LOW', 'COVER_NONE', 'BOTTLE_SMASHED', 'LABEL_WHITE', 'LABEL_MISPLACED', 'LABEL_NONE', 'BOTTLE_NONE']] = predictions
submission.to_csv('submission.csv', index=False)

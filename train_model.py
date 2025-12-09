import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# --- CONFIGURAÇÕES ---
DATASET_DIR = 'dataset_lixo'  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
MODEL_FILENAME = 'meu_modelo_ecoia.h5'

# 1. Verificação de segurança
if not os.path.exists(DATASET_DIR):
    print(f"ERRO: A pasta '{DATASET_DIR}' não existe.")
    exit()

print("--- Preparando as imagens ---")

# 2. Configura o gerador
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(f"\nCLASSES: {class_names}")

# 3. Cria o Modelo
print("\n--- Baixando MobileNetV2 ---")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Treina (A CORREÇÃO ESTÁ AQUI)
print(f"\n--- Iniciando Treinamento ({EPOCHS} épocas) ---")

history = model.fit(
    train_generator,
    # steps_per_epoch removido ou substituído por len() para evitar erro de fim de dados
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=EPOCHS
)

# 5. Salva
print("\n--- Salvando modelo ---")
model.save(MODEL_FILENAME)
print(f"SUCESSO! Modelo salvo: {MODEL_FILENAME}")
print("-" * 50)
print(train_generator.class_indices)
print("-" * 50)
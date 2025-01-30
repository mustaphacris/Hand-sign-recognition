import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import Counter
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback 


# Chemin vers le dossier contenant les sous-dossiers des signes
DATASET_DIR = os.path.expanduser("C:/Users/Musta/OneDrive/Bureau/projet_imagerie/test")
AUGMENTED_DATASET_DIR = os.path.expanduser("C:/Users/Musta/OneDrive/Bureau/projet_imagerie/augmented_data")
LANDMARKS_DIR = os.path.expanduser("C:/Users/Musta/OneDrive/Bureau/projet_imagerie/landmarks")

#MODEL_PATH = os.path.join(OUTPUT_DIR, "sign_language_model.keras")
IMG_SIZE = (256, 256)  # Taille des images



os.makedirs(AUGMENTED_DATASET_DIR, exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)

# Récupérer les classes dynamiquement
SIGN_CLASSES = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

# Fonction pour augmenter les données et les sauvegarder
def augment_data(input_dir, output_dir, img_size, target_count=300):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for class_name in SIGN_CLASSES:
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        images = []
        for img_name in os.listdir(class_input_path):
            img_path = os.path.join(class_input_path, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = load_img(img_path, target_size=img_size)
                #img_array = img_to_array(img)
                img_array = preprocess_input(img_to_array(img))

                images.append(img_array)

        images = np.array(images)
        images = images / 255.0  # Normalisation

        i = 0
        for batch in datagen.flow(images, batch_size=1, save_to_dir=class_output_path, save_prefix=class_name, save_format='jpg'):
            i += 1
            if i >= target_count - len(images):
                break

# Augmenter les données
augment_data(DATASET_DIR, AUGMENTED_DATASET_DIR, IMG_SIZE, target_count=300)

# Détection des landmarks dans les images augmentées
def detect_landmarks_on_dataset(input_dirs, output_dir):
    """
    Applique la détection des landmarks aux images réelles et augmentées.
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
 
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        for input_dir in input_dirs:  # On boucle sur les dossiers DATASET_DIR et AUGMENTED_DATASET_DIR
            for class_name in SIGN_CLASSES:
                class_input_path = os.path.join(input_dir, class_name)
                class_output_path = os.path.join(output_dir, class_name)
                os.makedirs(class_output_path, exist_ok=True)

                for img_name in os.listdir(class_input_path):
                    img_path = os.path.join(class_input_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image = cv2.imread(img_path)
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb_image)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            output_path = os.path.join(class_output_path, img_name)
                            cv2.imwrite(output_path, image)

# Exécuter la détection des landmarks pour les images réelles et augmentées
detect_landmarks_on_dataset([DATASET_DIR, AUGMENTED_DATASET_DIR], LANDMARKS_DIR)

print("Landmark detection completed and stored in landmarks directory.")


def balance_dataset(output_dir, img_size):
    """
    Équilibre le dataset en augmentant les classes sous-représentées.
    """
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Vérifier le nombre d'images par classe
    class_counts = {class_name: len(os.listdir(os.path.join(output_dir, class_name))) for class_name in SIGN_CLASSES}
    max_count = max(class_counts.values())  # Trouver la classe la plus représentée

    print("\n Nombre d'images par classe avant équilibrage :", class_counts)
    
    for class_name, count in class_counts.items():
        class_path = os.path.join(output_dir, class_name)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Si la classe a moins d'images que la classe majoritaire, on génère des images supplémentaires
        if count < max_count:
            images_to_generate = max_count - count
            print(f" Augmentation de la classe '{class_name}' ({count} → {max_count} images)")

            img_arrays = []
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                img_arrays.append(img_array)

            img_arrays = np.array(img_arrays) / 255.0  # Normalisation

            i = 0
            for batch in datagen.flow(img_arrays, batch_size=1, save_to_dir=class_path, save_prefix=class_name, save_format='jpg'):
                i += 1
                if i >= images_to_generate:
                    break  # Stopper dès que l'on a généré assez d'images
    
    print("\n Équilibrage du dataset terminé !")
 
# Équilibrage des classes après la génération des landmarks
balance_dataset(LANDMARKS_DIR, IMG_SIZE)

# Préparation des données pour l'entraînement
images, labels = [], []
for class_name in SIGN_CLASSES:
    class_path = os.path.join(LANDMARKS_DIR, class_name)
    label = SIGN_CLASSES.index(class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(label)

x_train, x_val, y_train, y_val = train_test_split(np.array(images), np.array(labels), test_size=0.2, random_state=42, stratify=labels)

# Encodage des labels en one-hot
y_train = to_categorical(y_train, num_classes=len(SIGN_CLASSES))
y_val = to_categorical(y_val, num_classes=len(SIGN_CLASSES))


print("Distribution des classes dans le train :", Counter(np.argmax(y_train, axis=1)))
print("Distribution des classes dans le val :", Counter(np.argmax(y_val, axis=1)))


# Création du modèle
# Charger MobileNetV2 en gardant les premières couches gelées
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Débloquer certaines couches du modèle pour l'entraînement
for layer in base_model.layers[-20:]:  # Débloquer les 20 dernières couches
    layer.trainable = True

base_model.trainable = False  # Ne pas entraîner les couches pré-entraînées

model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.4),  # Ajout de Dropout après MobileNetV2
    Dense(256, activation='relu'),
    Dropout(0.6),
    Dense(len(SIGN_CLASSES), activation='softmax')
])


# Compilation
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])


class StopTrainingCallback(Callback):
    def __init__(self, target_acc=0.98, target_loss=0.2):  
        super(StopTrainingCallback, self).__init__() 
        self.target_acc = target_acc
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get("accuracy")
        loss = logs.get("loss")

        if accuracy is not None and accuracy >= self.target_acc:
            print(f"\n Arrêt de l'entraînement : Accuracy {accuracy:.4f} a dépassé {self.target_acc*100}%")
            self.model.stop_training = True
        
        if loss is not None and loss <= self.target_loss:
            print(f"\n Arrêt de l'entraînement : Loss {loss:.4f} est en dessous de {self.target_loss}")
            self.model.stop_training = True

# Ajouter le callback personnalisé
stop_training = StopTrainingCallback(target_acc=0.98, target_loss=0.2)

# Entraînement
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=32, callbacks=[early_stopping, reduce_lr, stop_training]  # Ajout du callback personnalisé
)



# Tracer l'accuracy et la loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Sauvegarde du modèle
MODEL_PATH = os.path.join(LANDMARKS_DIR, "sign_language_model.keras")
model.save(MODEL_PATH)

print("Model training completed and saved.")

# Chargement du modèle entraîné
MODEL_PATH = os.path.join(LANDMARKS_DIR, "sign_language_model.keras")
model = load_model(MODEL_PATH)

# Détection et classification en temps réel
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Appuyez sur 'q' pour quitter")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, IMG_SIZE)
                    roi_normalized = roi_resized / 255.0
                    roi_expanded = np.expand_dims(roi_normalized, axis=0)

                    predictions = model.predict(roi_expanded)
                    predicted_label = np.argmax(predictions)
                    confidence = np.max(predictions)
                    predicted_class = SIGN_CLASSES[predicted_label]

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

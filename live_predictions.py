import keras
import librosa
import numpy as np
import os

from config import EXAMPLES_PATH
from config import MODEL_DIR_PATH

class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file
        self.model_path = None
        for model_file in ['Emotion_Voice_Detection_Model.h5', 'Emotion_Voice_Detection_Model.keras']:
            path = os.path.join(MODEL_DIR_PATH, model_file)
            if os.path.exists(path):
                self.model_path = path
                break
        if self.model_path is None:
            raise FileNotFoundError(f"Model file not found in {MODEL_DIR_PATH}")
        self.loaded_model = keras.models.load_model(self.model_path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=0)  # Add a new dimension for batch size
        x = np.expand_dims(x, axis=-1)     # Add a new dimension for channel
        predictions = self.loaded_model.predict(x)
        print("Raw model predictions:", predictions)  # Debug print statement
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class index:", predicted_class)  # Debug print statement
        print("Prediction is", " ", self.convert_class_to_emotion(predicted_class))

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}
        
        label = label_conversion.get(str(pred[0]), "Unknown emotion")
        return label

if __name__ == '__main__':
    try:
        live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, '03-01-01-01-01-02-05.wav'))
        live_prediction.loaded_model.summary()
        live_prediction.make_predictions()
        live_prediction = LivePredictions(file=os.path.join(EXAMPLES_PATH, '10-16-07-29-82-30-63.wav'))
        live_prediction.make_predictions()
    except FileNotFoundError as e:
        print(e)

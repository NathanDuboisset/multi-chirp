from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import data as data
import model as model




def evaluateModel(trained_model, test_data_path, batch_size):
    # Load data
    x_data, y_data, labels, file_paths = data.loadData(test_data_path)

    # Initialize AudioDataGenerator
    test_gen = model.AudioDataGenerator(file_paths, y_data, batch_size=batch_size)
    
    # Evaluate on the testdata
    loss, accuracy, precision, recall = trained_model.evaluate(test_gen)

    print("Test Loss: ", loss)
    print("Test Accuracy: ", accuracy)
    print("Test Precision: ", precision)
    print("Test Recall: ", recall)

    # Predictions on testdata
    y_pred = trained_model.predict(test_gen)

    # Classification report and confusion matrix
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.concatenate([np.array(batch[1]) for batch in test_gen])
    y_true_binary = y_true.argmax(axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_binary, y_pred_classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_binary, y_pred_classes))



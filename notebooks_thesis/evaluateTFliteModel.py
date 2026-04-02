import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import data
import model

def evaluate_tflite_model(tflite_model_path, test_data_path, batch_size):
    # Load data
    x_test, y_test, labels, file_paths = data.loadData(test_data_path)
    
    # balance the testing data:
    print('Balance the test data...')

    y_test_indices = np.argmax(y_test, axis=1)

    # minimum of one class
    min_samples = min(np.bincount(y_test_indices))

    # reduce entries until minimum after shuffle 
    balanced_x_test = []
    balanced_y_test = []
    balanced_file_paths_test = []
    for label in np.unique(y_test_indices):
        indices = np.where(y_test_indices == label)[0]
        np.random.shuffle(indices)  # Random order for random removal of samples
        indices = indices[:min_samples]
        balanced_x_test.append(x_test[indices])
        balanced_y_test.append(y_test[indices])
        balanced_file_paths_test.extend(file_paths[indices])

    # Combine the balanced data for all classes
    balanced_x_test = np.concatenate(balanced_x_test, axis=0)
    balanced_y_test = np.concatenate(balanced_y_test, axis=0)
    balanced_file_paths_test = np.array(balanced_file_paths_test)

    print('Balanced test data:')
    print('balanced_x_test shape:', balanced_x_test.shape)
    print('balanced_y_test shape:', balanced_y_test.shape)
    print('balanced_file_paths_test shape:', balanced_file_paths_test.shape)

    print('...Done. Loaded {} test samples and {} labels.'.format(balanced_x_test.shape[0], balanced_y_test.shape[0]), flush=True)





    # Initialize AudioDataGenerator
    test_gen = model.AudioDataGenerator(balanced_file_paths_test, balanced_y_test, batch_size=batch_size)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize variables for confusion matrix
    y_true = []
    y_pred = []

    # Iterate over test data batches
    for i, (batch_x, batch_y) in enumerate(test_gen):
        
        if i % 10 == 0:
            # Print progress after every 10 batch
            print("Processing batch", i+1, "of", len(test_gen))

        # Extract the true labels from the batch
        true_labels = np.argmax(batch_y, axis=1)

        # Iterate over files in the batch
        for j, file_x in enumerate(batch_x):
            # Reshape the audio file to match the expected input shape
            input_data = np.expand_dims(file_x, axis=0)
#DEBUG---   
            # get input file path
            file_path = balanced_file_paths_test[len(y_true) - 1]
            # Print the file path

#---
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Postprocess output data if needed
            predicted_labels = np.argmax(output_data, axis=1)

            # Append true and predicted labels for confusion matrix
            y_true.append(true_labels[j])
            y_pred.append(predicted_labels[0])
            
 #DEBUG     # Print every 100th entry
            if len(y_true) % 100 == 0:
                print("Input File Path:", file_path)
                print("Entry:", len(y_true))
                print("True Label:", y_true[-1])
                print("Predicted Label:", y_pred[-1])
                print("Output Tensor:", output_data)
#---
    # Calculate accuracy
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    print("Accuracy: {:.2%}".format(accuracy))

    # Calculate recall
    recall = np.sum(np.diag(confusion_matrix(y_true, y_pred))) / np.sum(np.array(y_true))
    print("Recall: {:.2%}".format(recall))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    target_names = [labels[i] for i in range(len(labels))]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
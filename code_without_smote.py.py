import itertools
import os
import numpy as np
import pandas as pd
import mne
import warnings
import time
import cv2
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D, Input, Flatten
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import shap
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds 
np.random.seed(42)
tf.random.set_seed(42)

# Load the Dataset
def load_data(data_path, tasks, n_subs, n_sessions):
    x, y = [], []
    for sub_n, session_n in itertools.product(range(n_subs), range(n_sessions)):
        epochs_data, labels = [], []
        for lab_idx, level in enumerate(tasks):
            sub = 'P{0:02d}'.format(sub_n + 1)
            sess = f'S{session_n + 1}'
            path = os.path.join(os.path.join(data_path, sub), sess) + f'/eeg/alldata_sbj{str(sub_n + 1).zfill(2)}_sess{session_n + 1}_{level}.set'
            epochs = mne.io.read_epochs_eeglab(path, verbose=False)
            epochs.pick_channels(channel_names)
            tmp = epochs.get_data()
            epochs_data.extend(tmp)
            labels.extend([lab_idx] * len(tmp))
        x.extend(epochs_data)
        y.extend(labels)
    return np.array(x), np.array(y)


 

# EEGNet Model Definition
def build_eegnet(nb_classes, Chans=61, Samples=500, dropoutRate=0.5, kernLength=64, F1=64, D=4, F2=128):
    input1 = Input(shape=(Chans, Samples, 1))
    
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

# Train the Model
def train_model(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=50):
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )
    return history

# Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))
    
    return precision, recall, f1

# Plot and Save Accuracy and Loss Curves
def plot_accuracy_loss_curves(history, trial):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Trial {trial + 1} - Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Trial {trial + 1} - Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'Trial_{trial + 1}_Accuracy_Loss_Curves.png')
    plt.close()

# Plot and Save Confusion Matrix
def plot_confusion_matrix(model, X_test, y_test, trial):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Trial {trial + 1} - Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true_classes)))
    plt.xticks(tick_marks, np.unique(y_true_classes))
    plt.yticks(tick_marks, np.unique(y_true_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    plt.savefig(f'Trial_{trial + 1}_Confusion_Matrix.png')
    plt.close()


def save_model_summary_as_text(model, filename="model_summary.txt"):
    """Saves model summary to a text file."""
    with open(filename, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved as {filename}")

def convert_text_to_image(text_file="model_summary.txt", image_file="model_summary_image.png"):
    """Converts text file content into an image."""
    with open(text_file, "r") as f:
        text = f.readlines()
    
    # Image parameters
    font_scale = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_spacing = 20
    padding = 30

    # Calculate image height dynamically based on text length
    img_width = 800
    img_height = padding + len(text) * line_spacing
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # White background

    # Write text to image
    y_offset = padding
    for line in text:
        cv2.putText(img, line.strip(), (padding, y_offset), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        y_offset += line_spacing

    # Save the image
    cv2.imwrite(image_file, img)
    print(f"Model summary image saved as {image_file}")

def save_model_summary_as_image(model):
    """Saves model architecture as an image using plot_model."""
    plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
    print("Model architecture saved as model_architecture.png")

# SHAP Analysis Function
def shap_analysis(model, X_train, X_test, output_dir, sample_size=50):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(output_dir)}")

    # Explain the model's predictions using SHAP
    explainer = shap.GradientExplainer(model, X_train[:sample_size])
    shap_values = explainer.shap_values(X_test[:sample_size])
    
    # Debug: Check SHAP values shape
    print("SHAP values shape:", np.array(shap_values).shape)

    # Aggregate SHAP values across the time dimension
    shap_values_aggregated = [np.mean(shap_values[class_idx], axis=2) for class_idx in range(len(shap_values))]  # Shape: (3, 50, 11)
    shap_values_aggregated = [np.squeeze(shap_values_aggregated[class_idx]) for class_idx in range(len(shap_values_aggregated))]  # Remove the last dimension

    # Aggregate X_test across the time dimension
    X_test_aggregated = np.mean(X_test[:sample_size], axis=2)  # Shape: (50, 11, 1)
    X_test_aggregated = np.squeeze(X_test_aggregated)  # Remove the last dimension

    # Save SHAP summary plot for each class
    print("Saving SHAP summary plot...")
    for class_idx in range(len(shap_values_aggregated)):
        # Select SHAP values for the current class
        shap_values_class = shap_values_aggregated[class_idx]  # Shape: (50, 11)

        # Plot SHAP summary for the current class
        shap.summary_plot(shap_values_class, X_test_aggregated, plot_type="bar", feature_names=channel_names,max_display=61, show=False)
        plt.savefig(os.path.join(output_dir, f"shap_summary_plot_class_{class_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    print("SHAP summary plots saved.")

    # Save SHAP image plot
    print("Saving SHAP image plot...")
    plt.figure(figsize=(14, 12))  # Set figure size
    shap.image_plot(shap_values, X_test[:sample_size], show=False)
    plt.savefig(os.path.join(output_dir, "shap_image_plot.svg"), bbox_inches='tight')
    plt.close()
    print("SHAP image plot saved.")

    # Save SHAP heatmap for a sample
    print("Saving SHAP heatmap...")
    sample_idx = 9  # Choose a sample index
    shap_values_np = np.array(shap_values)
    if len(shap_values_np.shape) == 5:
        shap_values_sample = shap_values_np[0, sample_idx, :, :, 0]  # Select first class
    else:
        shap_values_sample = shap_values_np[sample_idx, :, :, 0]

    plt.imshow(shap_values_sample, aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Channels")
    plt.title(f"SHAP Heatmap for Sample {sample_idx}")
    plt.savefig(os.path.join(output_dir, "shap_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP heatmap saved.")

    # Save SHAP dependence plot for each feature
    print("Saving SHAP dependence plots...")
    for feature_idx in range(X_test_aggregated.shape[1]):
        shap.dependence_plot(
            feature_idx,
            shap_values_aggregated[0],  # Use SHAP values for the first class
            X_test_aggregated,
            feature_names=channel_names,
            show=False
        )
        plt.savefig(os.path.join(output_dir, f"shap_dependence_plot_feature_{feature_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    print("SHAP dependence plots saved.")

  

    # Save SHAP class comparison plot
    print("Saving SHAP class comparison plot...")
    shap.summary_plot(shap_values_aggregated, X_test_aggregated, plot_type="bar", feature_names=channel_names, max_display=61,show=False)
    plt.savefig(os.path.join(output_dir, "shap_class_comparison_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP class comparison plot saved.")

    # Save Global Feature Importance Plot
    print("Saving Global Feature Importance Plot...")

    # Aggregate SHAP values across time and samples
    shap_values_global = np.mean(np.abs(shap_values), axis=(1, 3))  # Shape: (3, 11)
    shap_values_global = np.squeeze(shap_values_global)  # Remove the last dimension

    # Plot global feature importance
    shap.summary_plot(shap_values_global, X_test_aggregated, plot_type="bar", feature_names=channel_names, max_display=61,show=False)
    plt.savefig(os.path.join(output_dir, "global_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Global Feature Importance Plot saved.")

    # Save Local Explanation Summary Plot
    print("Saving Local Explanation Summary Plot...")

    # Aggregate SHAP values across time and samples for the first class
    shap_values_local = np.mean(shap_values[0], axis=2)  # Shape: (50, 11)
    shap_values_local = np.squeeze(shap_values_local)  # Remove the last dimension

    # Plot local explanation summary for the first class
    shap.summary_plot(shap_values_local, X_test_aggregated, plot_type="dot", feature_names=channel_names, max_display=61,show=False)
    plt.savefig(os.path.join(output_dir, "local_explanation_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Local Explanation Summary Plot saved.")

    print("All SHAP plots saved successfully!")


# Main Workflow
if __name__ == "__main__":
    # Dataset Parameters
    data_path = 'Dataset_path'
    tasks = ['MATBeasy', 'MATBmed', 'MATBdiff']
    n_subs, n_sessions = 15, 2
    channel_names =['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'FCz', 'C4', 'T8', 'FT8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT10', 'F6', 'AF8', 'AF4','F2']
 
    # Load Data
    X, Y = load_data(data_path, tasks, n_subs, n_sessions)
    X = (X - np.mean(X, axis=(0, 2), keepdims=True)) / (np.std(X, axis=(0, 2), keepdims=True) + 1e-10)
    X, Y = shuffle(X, Y, random_state=42)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    Y = to_categorical(Y, len(np.unique(Y)))

    # Initialize DataFrame to store metrics
    results_df = pd.DataFrame(columns=['Trial', 'Train_Accuracy', 'Train_Loss', 'Validation_Accuracy', 'Validation_Loss', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'Training_Time'])

    # Run 5 trials
    for trial in range(5):
        print(f"\n=== Trial {trial + 1} ===")
       


        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        
        # Build and Compile Model
        model = build_eegnet(nb_classes=3)  # Ensure nb_classes matches the number of unique classes
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        # Measure training time
        start_time = time.time()
        
        # Train Model
        history = train_model(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=50)
        end_time = time.time()
        training_time = end_time - start_time
        
        # Evaluate Model
        train_accuracy = history.history['accuracy'][-1]
        train_loss = min(history.history['loss'])
    
        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = min(history.history['val_loss'])
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
        precision, recall, f1 = evaluate_model(model, X_test, y_test)
        
        # Save metrics
        results_df = results_df.append({
            'Trial': trial + 1,
            'Train_Accuracy': train_accuracy,
	    'Train_Loss': train_loss,
            'Validation_Accuracy': val_accuracy,
            'Validation_Loss': val_loss,
            'Test_Accuracy': test_accuracy,
            
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Training_Time': training_time
        }, ignore_index=True)
        
#Save the .h5 file
        model.save(f"EEGNet_Trial_{trial + 1}.h5")

        

 # Plot and save accuracy and loss curves
        plot_accuracy_loss_curves(history, trial)
        
        # Plot and save confusion matrix
        plot_confusion_matrix(model, X_test, y_test, trial)
        # Save SHAP plots for this trial
        shap_output_dir = f"Trial_{trial + 1}_SHAP_Plots"
        shap_analysis(model, X_train, X_test, shap_output_dir, sample_size=50)
    
    # Save results to Excel
    results_df.to_excel("Trial_Results.xlsx", index=False)
    print("Results saved to Trial_Results.xlsx")
    print(X.shape)
    print(Y.shape)
    

   # print("All trials completed and plots saved successfully!")
    plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
    print("Model architecture saved as model_architecture.png")
    save_model_summary_as_text(model)
    convert_text_to_image()
    save_model_summary_as_image(model)

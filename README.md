# 🧠 EarlyMed: Advanced Alzheimer's Detection System

Welcome to **EarlyMed**, a powerful and user-friendly tool designed to detect the early stages of **Alzheimer’s disease** from brain MRI scans using **deep learning**. Built with TensorFlow and Gradio, this project aims to assist both researchers and the general public in gaining insights into brain health through an AI-powered diagnosis interface.

---

## 🚀 Project Overview

This project uses a **Convolutional Neural Network (CNN)** trained on MRI images to classify brain scans into four categories:

- 🟢 **Non Demented**
- 🟡 **Very Mild Demented**
- 🟠 **Mild Demented**
- 🔴 **Moderate Demented**

It features a clean and interactive **Gradio UI** that allows users to upload MRI scans and receive instant predictions, along with stage-specific recommendations for next steps.

---

## 🏗️ Model Architecture

The CNN was trained using the Alzheimer's MRI dataset. Key components of the model include:

- Multiple **convolutional and pooling layers** for feature extraction
- **Dropout** for regularization
- A final **Dense softmax layer** for classification

The training pipeline included:

- Image normalization and resizing
- Training-validation split
- Use of `Adam` optimizer and `categorical_crossentropy` loss
- Metrics tracked: **accuracy** and **AUC**

### ✅ Evaluation Results:
- **Test Accuracy**: ~91%
- **AUC Score**: ~0.97
- Model saved as: `alzheimer_cnn_model.h5`

---

## 💡 Features

- ✅ **AI-Based MRI Scan Classification**
- 📊 **Real-Time Results with Confidence Scores**
- 📖 **Custom Recommendations Based on Diagnosis**
- 👨‍⚕️ **Stage-Specific Medical Advice**
- 🌐 **Easy Web Interface with Gradio**

---

## 🖼️ App Preview

![image](https://github.com/user-attachments/assets/6eb6c56f-eb1f-4af6-a10c-33f2a3c5e697)


---

## 🧪 How It Works

1. **User uploads an MRI brain scan** via Gradio.
2. Image is preprocessed (resized, normalized).
3. Image is passed to the trained **CNN model**.
4. The model returns a predicted class and confidence score.
5. Based on the prediction, a **personalized explanation** is provided.

---

## 🧰 Tech Stack

| Tech | Usage |
|------|-------|
| **TensorFlow / Keras** | CNN training and model loading |
| **NumPy** | Image array manipulation |
| **PIL (Pillow)** | Image preprocessing |
| **Gradio** | Interactive web-based UI |
| **Jupyter Notebook** | Model training and evaluation |

---

## 📁 File Structure

```
├── Memoraid_EarlyMed_Alzheimer's_Disease_Detection_Using_Deep_Learning   # Notebook for training and evaluating the model
├── Gradio App                       # Saved Keras model
  ├── app.py
  ├── README.md
  ├── alzheimer_cnn_model.h5
  ├── requirements.txt                       
├── README.md                         
```

---

## 🌍 Live Demo

Huggingface space: https://huggingface.co/spaces/MahatirTusher/EarlyMed_Alzeimer_Diagnosis

---

## ⚠️ Disclaimer

> We strongly urge users to consult a healthcare professional for appropriate medical guidance after getting the diagnosis. This initiative is developed by our team at VIT-AP University with the goal of empowering individuals to be more aware of their health before visiting a doctor. Our mission is to leverage AI for early detection and better healthcare awareness.
---

## 👩‍💻 Developed By

**Team EarlyMed**  
VIT-AP University  
_Mission: Early Detection, Smarter Decision._

---

## 📬 Feedback or Collaboration?

If you'd like to collaborate, improve the model, or deploy this to a medical center's early diagnostic suite — feel free to reach out via [email](mailto:mahatirtusher@gmail.com) or fork the repo and contribute!

---

Let me know if you'd like a version of this in Bengali too or want a license added (like MIT or Apache 2.0)!


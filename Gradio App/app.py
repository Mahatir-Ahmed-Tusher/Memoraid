import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
try:
    model = tf.keras.models.load_model("alzheimer_cnn_model.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Class labels
class_labels = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

# Stage descriptions for each class
stage_descriptions = {
    'Non Demented': """
    🟢 **Healthy Stage Assessment**
    * Your brain scan shows normal patterns with no significant signs of cognitive decline.
    * This is a positive indication of healthy brain function.
    
    **Recommended Actions:**
    * Continue maintaining a healthy lifestyle
    * Regular exercise and balanced diet
    * Stay mentally active with brain exercises
    * Schedule routine check-ups
    
    Remember: Prevention is better than cure! Keep up the good habits! 🌟
    """,
    
    'Very Mild Demented': """
    🟡 **Early Stage Assessment**
    * Your scan shows patterns consistent with very mild cognitive changes.
    * Early detection gives you the best chance for effective management.
    
    **Important Next Steps:**
    * Schedule an appointment with a neurologist
    * Start brain-stimulating activities
    * Consider lifestyle modifications
    * Join early-stage support groups
    
    Early intervention can make a significant difference in managing symptoms. Don't delay seeking professional advice! 🌟
    """,
    
    'Mild Demented': """
    🟠 **Middle Stage Assessment**
    * Your scan indicates patterns associated with mild cognitive decline.
    * This stage requires attention and proper medical supervision.
    
    **Crucial Actions Required:**
    * Immediate consultation with a specialist
    * Begin medication evaluation
    * Establish a care plan
    * Consider safety modifications at home
    * Join support groups for guidance
    
    Professional medical support is essential at this stage. You're not alone in this journey! 🌟
    """,
    
    'Moderate Demented': """
    🔴 **Late Stage Assessment**
    * Your scan shows patterns consistent with moderate cognitive decline.
    * Immediate medical attention and care planning are essential.
    
    **Urgent Actions Required:**
    * Seek immediate medical consultation
    * Establish comprehensive care planning
    * Consider full-time care options
    * Join support groups for families and caregivers
    * Implement safety measures at home
    
    Professional care and support systems are crucial at this stage. Help is available! 🌟
    """
}

def predict_alzheimer(image):
    if model is None:
        return "❌ Model is not loaded. Please check 'alzheimer_cnn_model.h5'."

    try:
        image = Image.fromarray(image)
        expected_channels = model.input_shape[-1]
        input_size = model.input_shape[1:3]

        if expected_channels == 1:
            image = image.convert("L")

        image = image.resize(input_size)
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if expected_channels == 1:
            img_array = np.expand_dims(img_array, axis=-1)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions) * 100

        predicted_class = class_labels[predicted_class_index]
        stage_info = stage_descriptions[predicted_class]

        result_text = f"""
        ## 🧠 Alzheimer's Disease Assessment Results
        
        ### Diagnosis
        **Predicted Condition:** {predicted_class}
        **Confidence Score:** {confidence_score:.2f}%
        
        ### Assessment Details
        {stage_info}
        """
        return result_text
    except Exception as e:
        return f"❌ Error in prediction: {e}"

# Create Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 EarlyMed: Advanced Alzheimer's Detection System
    
    Welcome to EarlyMed—an initiative by our team at VIT-AP University dedicated to empowering you with early health insights. 
    Leveraging AI for early detection, our mission is simple: "Early Detection, Smarter Decision." 
    Our Advanced Alzheimer's Detection System project is one of our key efforts to help you stay informed before visiting a doctor.
    """)
    
    with gr.Tabs():
        with gr.Tab("🔍 Diagnosis Tool"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="numpy",
                        label="Upload Brain MRI Scan",
                        elem_classes="image-upload"
                    )
                    submit_btn = gr.Button(
                        "🔍 Analyze Scan",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    diagnosis_output = gr.Markdown(
                        value="Upload an MRI scan to receive the analysis results..."
                    )

        with gr.Tab("ℹ️ About Alzheimer's"):
            gr.Markdown("""
            ### Understanding Alzheimer's Disease: A Friendly Guide
            
            🧠 **What is Alzheimer's Disease?**
            
            Alzheimer's disease is a progressive brain disorder that affects memory, thinking, and behavior. 
            It is the most common cause of dementia, a condition that leads to a decline in cognitive abilities. 
            Over time, Alzheimer's makes everyday activities challenging and affects a person's ability to live independently.
            
            While there is no cure yet, early detection and proper care can slow its progression and improve quality of life.
            """)

        with gr.Tab("🤖 How It Works"):
            gr.Markdown("""
            ### How Our AI-Powered Alzheimer's Detector Works
            
            🧠 **Welcome to Our AI-Powered Alzheimer's Diagnosis Tool!**
            
            We understand how important it is to get a quick and reliable assessment of brain health. 
            Our tool uses **cutting-edge AI technology** to analyze MRI scans and detect early signs of Alzheimer's disease. 
            Here's how it works in simple terms:
            
            🔍 **How Does It Work?**
            1. **MRI Scan Input**: You upload a brain MRI image.
            2. **Smart Image Processing**: Our AI model cleans and prepares the image for analysis.
            3. **Deep Learning Analysis**: A powerful **Convolutional Neural Network (CNN)** studies patterns in the scan, 
               just like how doctors look for early signs of Alzheimer's.
            4. **Prediction & Confidence Score**: The AI predicts the condition and provides a confidence score.
            
            ✅ **Why Is This Reliable?**
            * ✔️ **Trained on Real MRI Data**: Our model has learned from thousands of real MRI scans
            * ✔️ **AI-Powered Accuracy**: Deep learning allows for fast, consistent analysis
            * ✔️ **Easy to Use**: No need for complex medical knowledge
            * ✔️ **Early Detection Advantage**: Catching Alzheimer's early helps in better treatment
            """)

    gr.Markdown("""
    ---
    ### ⚠️ Important Disclaimer
    
    We strongly urge users to consult a healthcare professional for appropriate medical guidance after getting the diagnosis. 
    This initiative is developed by our team at VIT-AP University with the goal of empowering individuals to be more aware 
    of their health before visiting a doctor. Our mission is to leverage AI for early detection and better healthcare awareness.
    
    *Developed by the team at VIT-AP University*
    """)

    # Connect the button to the prediction function
    submit_btn.click(
        fn=predict_alzheimer,
        inputs=image_input,
        outputs=diagnosis_output
    )

if __name__ == "__main__":
    demo.launch()
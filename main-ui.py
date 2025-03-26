# medical_imaging_app.py
import os
import numpy as np
import gradio as gr
from PIL import Image as PILImage
import time


# Function to create the medical imaging app with a back button
def create_medical_app():
    # CSS for custom styling
    css = """
    .container {
        border-radius: 10px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .results-container {
        background-color: #f8f9fa;
        border-left: 4px solid #4285f4;
    }
    .app-header {
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .button-container {
        display: flex;
        gap: 10px;
    }
    footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.8em;
        color: #666;
    }
    #back_button {
        background-color: #1a384a;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
        transition: background-color 0.3s;
        margin-right: auto;
    }
    #back_button:hover {
        background-color: #0f2231;
    }
    """

    # Sample medical specialties for dropdown
    specialties = [
        "General Analysis",
        "Chest X-Ray",
        "Brain MRI",
        "Abdominal CT",
        "Mammography",
        "Bone X-Ray",
    ]

    def analyze_image(image, specialty, include_measurements):
        if image is None:
            return None, "Please upload an image to begin analysis."

        # Simulate processing delay
        time.sleep(1.5)

        try:
            # Get image dimensions for the report
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                width, height = image.size

            measurements_text = ""
            if include_measurements:
                measurements_text = f"""
    ### 📏 Measurements
    - Image dimensions: {width} × {height} pixels
    - Aspect ratio: {width / height:.2f}
    - Key region of interest identified
    """

            # Different response based on selected specialty
            findings = "No significant abnormalities detected."
            image_type = "X-ray"
            region = "Unspecified"

            if specialty == "Chest X-Ray":
                image_type = "X-ray"
                region = "Chest"
                findings = "Lungs appear clear. No consolidation, effusion, or pneumothorax. Heart size within normal limits."
            elif specialty == "Brain MRI":
                image_type = "MRI"
                region = "Brain"
                findings = "No evidence of acute infarction, hemorrhage or mass effect. Ventricles are normal in size and configuration."
            elif specialty == "Abdominal CT":
                image_type = "CT Scan"
                region = "Abdomen"
                findings = "Liver, spleen, pancreas, and kidneys appear normal. No free fluid or abnormal masses detected."

            response = f"""
    ### 📋 Analysis Results

    - **Image Type:** {image_type}
    - **Region:** {region}
    - **Specialty:** {specialty}
    - **Findings:** {findings}
    - **Impression:** Normal study. No acute findings.

    {measurements_text}

    ⚠ **Disclaimer:** This is an AI-generated analysis and should not be used for clinical decisions. Please consult a qualified medical professional for proper diagnosis.
    """

            # Create a processed version of the image (just a placeholder)
            # In a real app, you might highlight regions of interest
            processed_img = image

            return processed_img, response
        except Exception as e:
            return None, f"Analysis error: {str(e)}"

    # Function to clear all inputs and outputs
    def clear_all():
        return (
            None,
            "General Analysis",
            True,
            None,
            "Results will appear here after analysis.",
        )

    # Define the Gradio interface with improved design
    blocks = gr.Blocks(css=css, title="Medical Imaging Diagnosis Assistant")

    with blocks:
        with gr.Row(elem_classes="app-header"):
            back_button = gr.Button("← Back to HR GPT", elem_id="back_button")
            gr.HTML("""
                <div style="text-align: center; flex-grow: 1;">
                    <h1 style="margin-bottom: 5px;">🏥 Medical Imaging Diagnosis Assistant</h1>
                    <p style="color: #666;">Upload medical images for AI-assisted analysis</p>
                </div>
            """)
            # Empty div for spacing
            gr.HTML("<div></div>")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="container"):
                    gr.Markdown("### 📤 Upload")
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Medical Image",
                        elem_id="image_upload",
                        height=300,
                    )

                    specialty_dropdown = gr.Dropdown(
                        choices=specialties,
                        value="General Analysis",
                        label="Select Analysis Type",
                        info="Choose the type of analysis to perform",
                    )

                    measurements_checkbox = gr.Checkbox(
                        label="Include Image Measurements",
                        value=True,
                        info="Add detailed measurements to the report",
                    )

                    with gr.Row(elem_classes="button-container"):
                        analyze_button = gr.Button(
                            "🔍 Analyze Image", variant="primary"
                        )
                        clear_button = gr.Button("🗑️ Clear All", variant="secondary")

            with gr.Column(scale=1):
                with gr.Group(elem_classes="container results-container"):
                    gr.Markdown("### 🔬 Results")
                    output_image = gr.Image(label="Processed Image", visible=True)
                    output_text = gr.Markdown(
                        value="Results will appear here after analysis."
                    )

        # Sample images gallery
        with gr.Group(elem_classes="container"):
            gr.Markdown("### 📸 Sample Images")
            with gr.Row():
                gr.Examples(
                    examples=[
                        [
                            "https://img.freepik.com/free-photo/chest-xray-scan-result-medical-technology_53876-14315.jpg",
                            "Chest X-Ray",
                            True,
                        ],
                        [
                            "https://img.freepik.com/free-photo/mri-brain-scan-medical-anatomical-concept_53876-14303.jpg",
                            "Brain MRI",
                            True,
                        ],
                    ],
                    inputs=[image_input, specialty_dropdown, measurements_checkbox],
                    outputs=[output_image, output_text],
                    fn=analyze_image,
                    cache_examples=True,
                )

        # Add information section at the bottom
        with gr.Group(elem_classes="container"):
            gr.Markdown("""
            ### ℹ️ About This Tool
            
            This AI assistant helps with preliminary analysis of medical images. It can identify common imaging types and provide basic observations.
            
            **Features:**
            - Support for X-ray, MRI, CT, and other common medical imaging formats
            - Specialized analysis for different body regions
            - Basic measurements and region identification
            - Clear button to reset all inputs and results
            
            **Remember:** This tool is for educational and demonstration purposes only and should not replace professional medical advice.
            """)

        # Footer
        gr.HTML("""
        <footer>
            <p>© 2025 Medical Imaging Assistant | Created with Gradio | For demonstration purposes only</p>
        </footer>
        """)

        # Set up the event handlers
        analyze_button.click(
            fn=analyze_image,
            inputs=[image_input, specialty_dropdown, measurements_checkbox],
            outputs=[output_image, output_text],
            api_name="analyze",
        )

        # Clear button functionality
        clear_button.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                image_input,
                specialty_dropdown,
                measurements_checkbox,
                output_image,
                output_text,
            ],
            api_name="clear",
        )

        # Back button functionality - redirect to main app
        back_button.click(
            fn=lambda: None,
            inputs=None,
            outputs=None,
            # _js="() => { window.location.href='/'; }",
        )

    return blocks


# For standalone testing
if __name__ == "__main__":
    demo = create_medical_app()
    demo.launch()

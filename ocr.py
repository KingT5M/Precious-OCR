import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\T5M\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load pre-trained TROCR model and processor for handwritten text
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# ===========================
# Step 1: Preprocess PDF Pages and Extract Printed Text
# ===========================

def preprocess_image(image):
    """Preprocess the image for better OCR results."""
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2)
    image = ImageEnhance.Sharpness(image).enhance(2)
    image = image.filter(ImageFilter.MedianFilter())
    return image

def extract_text_from_image(image):
    """Extract printed text from an image using Tesseract OCR."""
    preprocessed_image = preprocess_image(image)
    custom_config = r'--oem 1 --psm 6'
    extracted_text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return extracted_text

def extract_text_from_pdf(pdf_path):
    """Extract printed text from a PDF file using Tesseract OCR."""
    pages = convert_from_path(pdf_path, dpi=300)
    combined_text = ""
    for i, page in enumerate(pages):
        print(f"Processing printed text from page {i + 1}...")
        page_text = extract_text_from_image(page)
        combined_text += page_text + "\n\n"
    return combined_text

# ===========================
# Step 2: Use PyMuPDF to Get Coordinates of Marked Handwritten Fields
# ===========================

def get_handwritten_field_boxes(pdf_path):
    """Use PyMuPDF to extract manually marked regions and their coordinates."""
    doc = fitz.open(pdf_path)
    field_boxes = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        annot_list = page.annots()  # Get annotations (including marked regions)
        page_fields = {}

        if annot_list:
            for annot in annot_list:
                rect = annot.rect  # Bounding box coordinates
                content = annot.info.get("content", f"Field_{len(page_fields)+1}")  # Optional: Use content or default naming
                print(f"Page {page_num+1}, Field: {content}, Coordinates: {rect}")  # Print the coordinates
                page_fields[content] = rect
            field_boxes.append(page_fields)

    return field_boxes

# ===========================
# Step 3: Crop and Extract Handwritten Text from Specified Fields
# ===========================

def crop_handwritten_section(image, crop_box):
    """Crop the handwritten section of the image using a bounding box."""
    return image.crop(crop_box)

def extract_handwritten_text_from_image(image):
    """Extract handwritten text from a cropped image using TROCR."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=100)  # Increase max_new_tokens
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def extract_handwritten_text_from_pdf(pdf_path, handwritten_field_boxes):
    """Extract handwritten text from specified sections of a PDF form."""
    pages = convert_from_path(pdf_path, dpi=300)
    handwritten_text_data = {}
    
    for i, page in enumerate(pages):
        print(f"Processing handwritten text from page {i + 1}...")
        
        if i < len(handwritten_field_boxes):
            page_fields = handwritten_field_boxes[i]  # Get the crop boxes for this page
            
            # Crop each section and extract handwritten text
            for field_name, crop_box in page_fields.items():
                cropped_image = crop_handwritten_section(page, (crop_box[0], crop_box[1], crop_box[2], crop_box[3]))
                handwritten_text = extract_handwritten_text_from_image(cropped_image)
                handwritten_text_data[field_name] = handwritten_text

    return handwritten_text_data

# ===========================
# Step 4: Hybrid Workflow for Forms with Handwritten Fields
# ===========================

def process_form_pdf(pdf_path):
    """
    Process a form PDF to extract both printed and handwritten text.
    
    Args:
    - pdf_path (str): Path to the PDF file containing the form.
    
    Returns:
    - dict: Combined results of printed and handwritten text.
    """
    # Step 1: Extract printed text from the PDF using Tesseract OCR
    printed_text_data = extract_text_from_pdf(pdf_path)
    
    # Step 2: Extract bounding boxes of handwritten fields using PyMuPDF
    handwritten_field_boxes = get_handwritten_field_boxes(pdf_path)
    
    # Step 3: Extract handwritten text from the specified fields
    handwritten_text_data = extract_handwritten_text_from_pdf(pdf_path, handwritten_field_boxes)
    
    # Combine the results
    combined_data = {
        "printed_text": printed_text_data,
        "handwritten_text": handwritten_text_data
    }

    return combined_data

# ===========================
# Step 5: Testing the Full Pipeline
# ===========================
pdf_path = r'C:\Users\T5M\Desktop\PRECIOUS OCR\Fekan Howell - Proposal Form signed and dated 11072024-output.pdf'

# Process the document
combined_data = process_form_pdf(pdf_path)

# Output the extracted data
print("Printed Text Data:")
print(combined_data['printed_text'])

print("\nHandwritten Text Data:")
for field, content in combined_data['handwritten_text'].items():
    print(f"{field}: {content}")

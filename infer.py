import torch
import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp
import os
import argparse

# Path to the model checkpoint
CHECKPOINT_PATH = 'unet_model.pth'

# Define the UnetPlusPlus model
def create_model():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",        # Encoder: ResNet34
        encoder_weights="imagenet",    # Pretrained weights
        in_channels=3,                 # Input channels
        classes=3                      # Number of output classes
    )

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match model input
        transforms.ToTensor(),         # Convert to Tensor
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Postprocess and save the output
def postprocess_and_save(output, save_path):
    # Convert output to a segmentation mask
    output_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    output_image = Image.fromarray((output_mask * 255).astype("uint8"))
    output_image.save(save_path)

# Load model checkpoint
def load_checkpoint(model, checkpoint_path):
    """
    Loads the model weights from a checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint, strict=False)  # Allow mismatched keys if needed
    return model

# Main inference function
def infer(image_path):
    # Load and prepare the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()  # Create the UnetPlusPlus model
    model = load_checkpoint(model, CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    # Preprocess input image
    input_tensor = preprocess_image(image_path).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Postprocess and save the result
    output_path = "output_image.png"  # Fixed output filename
    postprocess_and_save(output, output_path)
    print(f"Output saved to {output_path}")

# Command line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")

    args = parser.parse_args()

    # Ensure the input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: The file {args.image_path} does not exist.")
    else:
        # Use the image path provided by the user
        print(f"Running inference on {args.image_path}...")
        infer(args.image_path)

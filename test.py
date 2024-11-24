import torch
import segmentation_models_pytorch as smp

def test_unetplusplus_model_loading(checkpoint_path):
    # Step 1: Define the UnetPlusPlus model
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )
    
    # Step 2: Try loading the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)  # Use strict=False for mismatched keys
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Step 3: Run a test forward pass
    try:
        test_input = torch.randn(1, 3, 256, 256)  # Example input, adjust dimensions if needed
        output = model(test_input)
        print("Test forward pass successful!")
        print("Output shape:", output.shape)
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    # Use the uploaded checkpoint file
    checkpoint_path = "unet_model.pth"  # Replace with the actual path if different
    test_unetplusplus_model_loading(checkpoint_path)

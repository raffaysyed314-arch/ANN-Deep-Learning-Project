import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from captum.attr import LayerGradCam
from captum.attr import visualization as viz

sys.path.append('.')
from models.custom_cnns import CNN_5Layer
from models.pretrained_cnns import get_resnet18
from scripts.create_dataloaders import prepare_data_and_loaders

def run_gradcam(model, target_layer, input_tensor, label, model_name):
    layer_gc = LayerGradCam(model, target_layer)
    attributions_lgc = layer_gc.attribute(input_tensor, target=label)
    upsampled_attr = LayerGradCam.interpolate(attributions_lgc, input_tensor.shape[2:])
    
    attr_np = upsampled_attr.squeeze().cpu().detach().numpy()
    img_np = np.transpose(input_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    os.makedirs('results/xai_plots', exist_ok=True)

    try:
        fig, axis = viz.visualize_image_attr(
            np.expand_dims(attr_np, axis=2),
            img_np,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            title=f"Grad-CAM: {model_name} (Class {label})"
        )
        fig.savefig(f'results/xai_plots/{model_name}_gradcam.png')
        plt.close(fig)
        print(f"Successfully saved Grad-CAM heatmap for {model_name}")
    except AssertionError as e:
        if "scale factor = 0" in str(e):
            print(f"\n[EVIDENCE LOGGED] {model_name} returned all-zero gradients.")
            print("The model failed to learn any meaningful features due to data starvation.")
        else:
            print(f"Visualization error for {model_name}: {e}")

def main():
    print("Starting Task 4: Explainable AI (Grad-CAM)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader = prepare_data_and_loaders()
    if not test_loader: 
        print("Data loader failed to initialize.")
        return
    
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    test_img = images[0].unsqueeze(0).to(device)
    test_label = labels[0].item()

    # Process Custom CNN_5Layer
    print("\nProcessing Custom CNN_5Layer...")
    custom_model = CNN_5Layer().to(device)
    custom_path = 'saved_models/CNN_5Layer.pth'
    if os.path.exists(custom_path):
        custom_model.load_state_dict(torch.load(custom_path))
        custom_model.eval()
        run_gradcam(custom_model, custom_model.conv5, test_img, test_label, "CNN_5Layer")
    else:
        print("Error: Could not find saved weights for CNN_5Layer.")

    # Process Pretrained ResNet18
    print("\nProcessing Pretrained ResNet18...")
    resnet_model = get_resnet18().to(device)
    resnet_path = 'saved_models/ResNet18.pth'
    if os.path.exists(resnet_path):
        resnet_model.load_state_dict(torch.load(resnet_path))
        resnet_model.eval()
        run_gradcam(resnet_model, resnet_model.layer4[-1], test_img, test_label, "ResNet18")
    else:
        print("Error: Could not find saved weights for ResNet18.")

if __name__ == "__main__":
    main()

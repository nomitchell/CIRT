import torch
import torchvision
import torchvision.transforms as transforms
from models.cirt_model import CIRT_Model
import argparse
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIRT Feature Inversion Validation')
    parser.add_argument('--model-path', type=str, default='cirt_model.pth', help='path to trained CIRT model')
    parser.add_argument('--content-image', type=str, required=True, help='path to content image')
    parser.add_argument('--style-image', type=str, required=True, help='path to style image')
    parser.add_argument('--output-image', type=str, default='generated_image.png', help='path to save generated image')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIRT model
    model = CIRT_Model().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Load images
    content_image = Image.open(args.content_image).convert('RGB')
    content_image = transform(content_image).unsqueeze(0).to(device)
    style_image = Image.open(args.style_image).convert('RGB')
    style_image = transform(style_image).unsqueeze(0).to(device)

    # 1. Get feature targets
    with torch.no_grad():
        content_features = model.backbone(content_image)
        _, style_projection = model(style_image)

    # 2. The optimization loop
    gen_image = torch.randn_like(content_image, requires_grad=True)
    optimizer = torch.optim.Adam([gen_image], lr=0.1)

    for i in range(200):
        optimizer.zero_grad()
        
        gen_features = model.backbone(gen_image)
        _, gen_projection = model(gen_image)
        
        content_loss = torch.nn.functional.mse_loss(gen_features, content_features)
        style_loss = torch.nn.functional.mse_loss(gen_projection, style_projection)
        
        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

    # 3. Save and inspect the final `gen_image`.
    torchvision.utils.save_image(gen_image, args.output_image)

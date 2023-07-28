import torch
from torchsummary import summary
from PIL import Image
from torchvision import transforms

def deeplab(n_classes: int): 
    return torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=False, num_classes=n_classes)

model = deeplab(5)

# device = 'mps'
device = 'cpu'


input_image = Image.open('img.jpg')
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
print('input shape: ', input_batch.shape)

input_batch = input_batch.to(device)
model.to(device)

model.eval()

with torch.no_grad():
    out = model(input_batch)['out']
    print(out.shape)

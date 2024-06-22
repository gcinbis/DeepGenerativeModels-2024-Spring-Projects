from torchvision.models import vgg16

VGG = vgg16(pretrained=True).features[:16].eval()
for param in VGG.parameters():
    param.requires_grad = False
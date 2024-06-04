import clip
import torch
import torchvision.transforms as T

# clip feature extractor
class CLIP():
    # initialize feature extractor
    def __init__(self):
        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set clip model and preprocessing
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def image_features(self, image):
        # get the batch size
        batch_size = image.shape[0]

        # init preprocessed images tensor
        prepocessed = torch.empty((batch_size, 3, 224, 224)).to(self.device)
        
        # preporcess the images
        for i, img in enumerate(image):
            prepocessed[i] = self.preprocess(T.ToPILImage()(img)).unsqueeze(0).to(self.device)


        # return image features
        return self.model.encode_image(prepocessed).to(torch.float32).detach()
    
    def text_features(self, text):
        # tokenize text
        tokenized = clip.tokenize(text).to(self.device)

        # return text features
        return self.model.encode_text(tokenized).to(torch.float32).detach()
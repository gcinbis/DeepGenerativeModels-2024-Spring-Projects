import torch
import torchvision
from typing import Any, Tuple, Union, List
from enum import Enum
from torchvision.utils import flow_to_image
from torch.cuda.amp import autocast

class RAFTModelSize(Enum):
    SMALL = 'raft_small'
    LARGE = 'raft_large'

class RAFT:
    def __init__(self, model_size: RAFTModelSize = RAFTModelSize.SMALL, **kwargs: Any):
        """
        Initializes the RAFTWrapper with the specified model size.

        Args:
            model_size (RAFTModelSize): The size of the RAFT model to use.
            kwargs (Any): Additional arguments for the model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.raft_transforms = self._load_model(model_size, **kwargs)
        self.model.eval()
        self._original_size = None

    def half(self):
        """
        Converts the model to half precision.

        Returns:
            RAFT: The model in half precision.
        """
        self.model = self.model.half()
        return self

    def _load_model(self, model_size: RAFTModelSize, **kwargs: Any) -> torch.nn.Module:
        """
        Loads the specified model.

        Args:
            model_size (RAFTModelSize): The size of the model to use.
            kwargs (Any): Additional arguments for the model.

        Returns:
            torch.nn.Module: The loaded model.
        """
        if model_size == RAFTModelSize.SMALL:
            weights = torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT
            model = torchvision.models.optical_flow.raft_small(weights, **kwargs)
            transforms = weights.transforms()
        elif model_size == RAFTModelSize.LARGE:
            weights = torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
            model = torchvision.models.optical_flow.raft_large(weights, **kwargs)
            transforms = weights.transforms()
        else:
            raise ValueError(f"Model size {model_size} is not supported.")
        return model.to(self.device), transforms

    def forward(self, image1: Union[torch.Tensor, List[Any], Any], image2: Union[torch.Tensor, List[Any], Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a forward pass through the model.

        Args:
            image1 (Union[torch.Tensor, List[Any], Any]): The first image tensor, list of images, or single image.
            image2 (Union[torch.Tensor, List[Any], Any]): The second image tensor, list of images, or single image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The flow predictions.
        """
        image1 = self._prepare_input(image1)
        image2 = self._prepare_input(image2)

        image1, image2 = self.raft_transforms(image1, image2)

        image1, image2 = image1.to(self.device), image2.to(self.device)

        with torch.no_grad():
            with autocast():
                flow_predictions = self.model(image1, image2)
        return self.postprocess(flow_predictions)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Calls the forward method of the model.

        Args:
            *args (Any): The arguments to pass to the forward method.
            **kwds (Any): The keyword arguments to pass to the forward method.

        Returns:
            Any: The output of the forward method.
        """
        return self.forward(*args, **kwds)

    def _prepare_input(self, image: Union[torch.Tensor, List[Any], Any]) -> torch.Tensor:
        """
        Prepares the input images for processing.

        Args:
            image (Union[torch.Tensor, List[Any], Any]): The input image(s).

        Returns:
            torch.Tensor: The prepared image tensor.
        """
        if isinstance(image, list):
            self._original_size = image[0].shape
            if not all(isinstance(img, torch.Tensor) for img in image):
                image = [self._apply_transform(img) for img in image]
            image = torch.stack(image)
        elif not isinstance(image, torch.Tensor):
            self._original_size = image.size
            image = self._apply_transform(image)
            image = image.unsqueeze(0)
        elif len(image.shape) == 3:
            self._original_size = image.shape
            image = image.unsqueeze(0)
            image = self._apply_transform(image)
        elif len(image.shape) == 4:
            self._original_size = image[0].shape
            image = self._apply_transform(image)
        return image

    def _apply_transform(self, image: Any) -> torch.Tensor:
        """
        Applies the transformation to the input image.

        Args:
            image (Any): The input image.

        Returns:
            torch.Tensor: The transformed image.
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((520, 960), antialias=False),
        ])
        if isinstance(image, torch.Tensor):
            transform.transforms = transform.transforms[1:]
        return transform(image)

    def postprocess(self, flow: torch.Tensor) -> Any:
        """
        Postprocesses the flow output.

        Args:
            flow (torch.Tensor): The flow output.

        Returns:
            Any: The postprocessed flow.
        """
        # Return np array
        return flow[-1].cpu().numpy().squeeze(0)
# Usage example
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Create example single image
    single_image1 = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    single_image2 = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))

    # Create example batched images
    batched_image1 = np.random.rand(2, 3, 256, 256).astype(np.float32)
    batched_image2 = np.random.rand(2, 3, 256, 256).astype(np.float32)
    
    batched_image1 = torch.tensor(batched_image1)
    batched_image2 = torch.tensor(batched_image2)

    # Initialize RAFT wrapper with model size choice
    raft = RAFT(model_size=RAFTModelSize.SMALL)

    # Get flow predictions for single images
    flow_predictions = raft(single_image1, single_image2)

    # Postprocess and display flow for single images
    flow = raft.postprocess(flow_predictions)
    print(flow)

    # Get flow predictions for batched images
    flow_predictions = raft(batched_image1, batched_image2)

    # Postprocess and display flow for batched images
    flow = raft.postprocess(flow_predictions)
    print(flow)

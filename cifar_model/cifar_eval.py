import torch
import torchvision
import torchvision.transforms as transforms
from typing import Optional
import numpy as np
from time import time


from cifar_model.torch_model import Net
from tensor_algos.tensor_I_algorithms import TensorAlgo
from tensor_algos.cp_als import CpAls



DOWNLOAD = False
PATH = './cifar_net.pth'

class CifarModelEvaluator():

    def __init__(self, sub_sample=1):
        self.path = PATH
        self.net = Net()
        self.net.load_state_dict(torch.load(self.path, weights_only=True))

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = 32


        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=DOWNLOAD, transform=self.transform)
        subset_data = torch.utils.data.Subset(testset, list(range(0,len(testset), sub_sample)))

        self.testloader = torch.utils.data.DataLoader(subset_data, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=4)


    def  _convert_images(self, images: torch.tensor, tensor_algo:  Optional[TensorAlgo]=None) -> torch.tensor:
        """
        Convert the images according to the algo.
        :param images: A batch of images to convert to numpy and run the algo upon
        :param tensor_algo: The tensors algo to run
        :return: A decomposed and composed image batch
        """
        if tensor_algo is not None:
            np_images = images.numpy()
            new_images = []
            for i in range(np_images.shape[0]):
                curr_img = np_images[i]
                converted_img = tensor_algo.compose(tensor_algo.decompose(curr_img))
                new_images.append(converted_img)
            stacked_new_images = np.stack(new_images, axis=0)
            images = torch.Tensor(stacked_new_images)
        return images

    def eval_model(self, tensor_algo: Optional[TensorAlgo]=None) -> float:
        """
        Evaluates the current model, when
        :param tensor_algo: If given, runs the decompostion algo and then composes it before entering the data
        into the model
        :return: The accuracy of the model
        """
        print(f"Evaluating {tensor_algo}")
        correct = 0
        total = 0
        all_frames = len(self.testloader)
        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                if tensor_algo is not None and i % 2 == 0:
                    print(f"Batch: {i}/{all_frames}")
                images, labels = data
                images = self._convert_images(images, tensor_algo)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

if __name__ == "__main__":
    decomp = CpAls(rank=3)
    evaler = CifarModelEvaluator(sub_sample=20)
    score = evaler.eval_model()
    score_composed = evaler.eval_model(tensor_algo=decomp)
    print(score, score_composed)
import os
import torch
from torch_helper import ClassifierTrainer, create_mnist_data_loaders, SeqModel
from torch import Tensor, nn
from torch_helper.data_loader import iter_batches, DataLoader

train_data_loader, test_data_loader = create_mnist_data_loaders()


class MnistModel(SeqModel):
    def __init__(self):
        seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
        )
        super().__init__(seq)


def save_state(model: nn.Module, path: str):
    state_dict = model.state_dict()
    torch.save(state_dict, path)


def load_state(model: nn.Module, path: str):
    if not os.path.exists(path):
        return
    state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(state_dict)


SAVE_PATH = "model.pth"


def calc_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    total_preds = 0
    correct_preds = 0

    with torch.no_grad():
        for x, y in iter_batches(data_loader):
            pred_y = model(x)
            correct_preds += calc_correct_preds(pred_y, y)
            total_preds += len(x)

    return correct_preds / total_preds


def calc_correct_preds(preds: Tensor, targets: Tensor) -> int:
    _, preds = preds.max(dim=1)
    return (preds == targets).sum().item()


class MyTrainer(ClassifierTrainer):
    def save(self):
        save_state(self._model, SAVE_PATH)

    def load(self):
        load_state(self._model, SAVE_PATH)


model = MnistModel()
trainer = MyTrainer(model)


def main():
    trainer.fit(train_data_loader, test_data_loader, epochs_count=30)


def test_accuracy():
    acc = calc_accuracy(trainer._model, test_data_loader)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    test_accuracy()
    # main()

# TODO: log train accuracy

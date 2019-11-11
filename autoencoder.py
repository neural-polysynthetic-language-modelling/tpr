import argparse
import configargparse
import logging
from typing import List, NamedTuple, Tuple

import torch  # type: ignore
import torch.nn  # type: ignore
from torch.nn.functional import relu  # type: ignore
import torch.optim  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

from corpus import MorphemeCorpus
from loss import UnbindingLoss
from morpheme import Morpheme
import util


class MorphemeVectors(torch.nn.Module):

    def __init__(self, *, corpus: MorphemeCorpus, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        super().__init__()

        self.device = device
        self.unbinding_loss = UnbindingLoss(alphabet=corpus.morphemes.alphabet, device=device)
        self.corpus: MorphemeCorpus = corpus
        self.input_dimension_size: int = corpus.morphemes.flattened_tpr_size
        self.hidden_layer_size: int = hidden_layer_size
        self.hidden_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        for n in range(num_hidden_layers):  # type: int
            if n == 0:
                self.hidden_layers.append(torch.nn.Linear(self.input_dimension_size,
                                                          self.hidden_layer_size,
                                                          bias=True))
            else:
                self.hidden_layers.append(torch.nn.Linear(self.hidden_layer_size,
                                                          self.hidden_layer_size,
                                                          bias=True))

        self.output_layer: torch.nn.Module = torch.nn.Linear(self.hidden_layer_size,
                                                             self.input_dimension_size,
                                                             bias=True)
        self.to(device=device)

    def to(self, device):
        super().to(device)
        self.device = device

    def forward(self, morphemes: List[Morpheme]) -> torch.Tensor:
        batch_size: int = len(morphemes)
        morpheme_tprs: torch.Tensor = MorphemeCorpus.collate_tprs(morphemes, self.device)
        tensor_at_input_layer: torch.Tensor = morpheme_tprs.view(batch_size, self.input_dimension_size)
        tensor_at_final_hidden_layer: torch.Tensor = self._apply_hidden_layers(tensor_at_input_layer)
        tensor_at_output_layer: torch.Tensor = self._apply_output_layer(tensor_at_final_hidden_layer)
        return tensor_at_output_layer.view(morpheme_tprs.shape)

    def _apply_hidden_layers(self, tensor_at_input_layer: torch.Tensor) -> torch.Tensor:
        tensor_at_previous_layer: torch.nn.Module = tensor_at_input_layer

        for hidden in iter(self.hidden_layers):  # type: torch.nn.Module
            tensor_before_activation: torch.Tensor = hidden(tensor_at_previous_layer)
            tensor_at_current_layer: torch.Tensor = relu(tensor_before_activation)
            tensor_at_previous_layer = tensor_at_current_layer

        return tensor_at_current_layer

    def _apply_output_layer(self, tensor_at_hidden_layer: torch.Tensor) -> torch.Tensor:
        return self.output_layer(tensor_at_hidden_layer)  # .cuda(device=cuda_device))

    @staticmethod
    def collate_morphemes(batch: List[Morpheme]) -> List[Morpheme]:
        return batch

    def evaluate(self, morphemes: List[Morpheme]) -> List[Morpheme]:
        tprs: torch.Tensor = self(morphemes)
        return self.unbinding_loss.unbind(tprs)

    def run_testing(self, *, batch_size: int) -> None:

        data_loader: DataLoader = DataLoader(dataset=self.corpus, batch_size=batch_size, shuffle=False,
                                             collate_fn=MorphemeVectors.collate_morphemes)

        for morphemes in iter(data_loader):  # type: List[Morpheme]
            predicted_morphemes: List[Morpheme] = self.evaluate(morphemes)
            for i in range(len(morphemes)):
                print(f"{morphemes[i]}\t{predicted_morphemes[i]}")

    def run_training(self, *,
                     learning_rate: float,
                     epochs: int,
                     batch_size: int,
                     logging_frequency: int) -> None:

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        data_loader: DataLoader = DataLoader(dataset=self.corpus, batch_size=batch_size, shuffle=True,
                                             collate_fn=MorphemeVectors.collate_morphemes)

        for epoch in range(1, epochs+1):

            optimizer.zero_grad()
            total_loss: float = 0.0

            for batch_number, morphemes in enumerate(data_loader):  # type: Tuple[int, List[Morpheme]]
                predictions: torch.Tensor = self(morphemes)
                labels: torch.Tensor = MorphemeCorpus.collate_tprs(morphemes, self.device)
                loss: torch.Tensor = self.unbinding_loss(predictions, labels)
                total_loss += loss.item()
                loss.backward()

            if epoch == 1:  # Report total loss before any optimization as loss at Epoch 0
                logging.info(f"Epoch {str(0).zfill(len(str(epochs)))}\ttrain loss: {total_loss}")
            elif epoch % logging_frequency == 0:
                logging.info(f"Epoch {str(epoch).zfill(len(str(epochs)))}\ttrain loss: {total_loss}")

            optimizer.step()


def configure_training(args: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser()
    p.add('-c', '--config', required=False, is_config_file=True, help='configuration file')

    p.add('--corpus', required=True, help='Pickle file containing a MorphemeCorpus object')
    p.add('--hidden_size', required=True, type=int)
    p.add('--hidden_layers', required=True, type=int)

    p.add('-o', '--output_file', required=True, type=str, metavar="FILENAME",
          help="Output file where trained MorphemeVectors model will be saved")

    p.add('--continue_training', required=False, type=bool, help='Continue training')

    p.add('--print_every', required=True, type=int)
    p.add('--batch_size', required=True, type=int)
    p.add('--num_epochs', required=True, type=int)
    p.add('--learning_rate', required=True, type=float)

    return p.parse_args(args=args)


def configure_testing(args: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser()
    p.add('-c', '--config', required=False, is_config_file=True, help='configuration file')

    p.add('--morpheme_vectors', required=True, help='Pickle file containing a MorphemeVectors object')
    p.add('--batch_size', required=True, type=int)

    return p.parse_args(args=args)


def evaluate(args: argparse.Namespace) -> None:
    device = util.get_device()
    model: MorphemeVectors = torch.load(args.morpheme_vectors)
    model.to(device)
    model.run_testing(args.batch_size)


def train(args: argparse.Namespace) -> None:

    device = util.get_device()

    logging.info(f"Training MorphemeVectors on {str(device)} using {args.corpus} as training data")

    model: MorphemeVectors = MorphemeVectors(
        corpus=MorphemeCorpus.load(args.corpus),
        hidden_layer_size=args.hidden_size,
        num_hidden_layers=args.hidden_layers,
        device=device)

    model.run_training(learning_rate=args.learning_rate,
                       epochs=args.num_epochs,
                       batch_size=args.batch_size,
                       logging_frequency=args.print_every)

    logging.info(f"Saving model to {args.output_file}")
    torch.save(model.to(torch.device("cpu")), args.output_file)


if __name__ == "__main__":

    import sys

    logging.basicConfig(
        level='INFO',
        stream=sys.stderr,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s\t%(message)s",
    )

    if '--morpheme_vectors' in sys.argv:
        evaluate(configure_testing(args=sys.argv[1:]))
    else:
        train(configure_training(args=sys.argv[1:]))
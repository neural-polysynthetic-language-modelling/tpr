import argparse
import configargparse
import logging
import sys
from typing import List

import torch  # type: ignore
import torch.nn  # type: ignore
from torch.nn.functional import relu  # type: ignore
import torch.optim  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

from corpus import MorphemeCorpus
from loss import UnbindingLoss
from morpheme import Morpheme


class MorphemeVectors(torch.nn.Module):

    def __init__(self, corpus: MorphemeCorpus, hidden_layer_size: int, num_hidden_layers: int):
        super().__init__()

        self.loss_function = UnbindingLoss(alphabet=corpus.morphemes.alphabet)
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

    def forward(self, morphemes: List[Morpheme]) -> torch.Tensor:
        morpheme_tprs: torch.Tensor = MorphemeCorpus.collate_tprs(morphemes)
        batch_size: int = len(morphemes)
        tensor_at_input_layer: torch.Tensor = morpheme_tprs.view(batch_size, self.input_dimension_size)
        tensor_at_final_hidden_layer: torch.Tensor = self._apply_hidden_layers(tensor_at_input_layer)
        tensor_at_output_layer: torch.Tensor = self._apply_output_layer(tensor_at_final_hidden_layer)
        return tensor_at_output_layer.view(morpheme_tprs.shape)

    def _apply_hidden_layers(self, tensor_at_input_layer: torch.Tensor) -> torch.Tensor:
        tensor_at_previous_layer: torch.nn.Module = tensor_at_input_layer

        for hidden in self.hidden_layers:  # type: torch.nn.Module
            tensor_at_current_layer: torch.Tensor = relu(hidden(tensor_at_previous_layer))
            tensor_at_previous_layer = tensor_at_current_layer

        return tensor_at_current_layer

    def _apply_output_layer(self, tensor_at_hidden_layer: torch.Tensor) -> torch.Tensor:
        return self.output_layer(tensor_at_hidden_layer)  # .cuda(device=cuda_device))

    def run_training(self, *,
                     # device: torch.device,
                     learning_rate: float,
                     epochs: int,
                     batch_size: int,
                     logging_frequency: int) -> None:

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        data_loader: DataLoader = DataLoader(dataset=self.corpus, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs+1):

            optimizer.zero_grad()
            total_loss: float = 0.0

            for batch_number, morphemes in enumerate(data_loader):  # type: Tuple[int, List[Morpheme]]
                predictions: torch.Tensor = self(morphemes)
                labels: torch.Tensor = MorphemeCorpus.collate_tprs(morphemes)
                loss: torch.Tensor = self.loss_function(predictions, labels)
                total_loss += loss.item()
                loss.backward()

            if epoch == 1:  # Report total loss before any optimization as loss at Epoch 0
                logging.info(f"Epoch {str(0).zfill(len(str(epochs)))}\ttrain loss: {total_loss}")
            elif epoch % logging_frequency == 0:
                logging.info(f"Epoch {str(epoch).zfill(len(str(epochs)))}\ttrain loss: {total_loss}")

            optimizer.step()


def configure(args: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser()
    p.add('-c', '--config', required=True, is_config_file=True, help='configuration file')

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


def train(args: argparse.Namespace) -> None:

    logging.info(f"Training MorphemeVectors using {args.corpus} as training data")

    model: MorphemeVectors = MorphemeVectors(
        corpus=MorphemeCorpus.load(args.corpus),
        hidden_layer_size=args.hidden_layer_size,
        num_hidden_layers=args.hidden_layers,
    )

    model.run_training(learning_rate=args.learning_rate,
                       epochs=args.num_epochs,
                       batch_size=args.batch_size,
                       output_filename=args.output_file,
                       logging_frequency=args.print_every,

    )

    logging.info(f"Saving model to {output_filename}")
    torch.save(model, args.output_file)


if __name__ == "__main__":

    import sys

    logging.basicConfig(
        level='INFO',
        stream=sys.stderr,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s\t%(message)s",
    )

    train(configure(arguments=sys.argv[1:]))
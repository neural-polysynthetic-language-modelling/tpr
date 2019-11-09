from typing import List, NamedTuple, Optional

import torch
from torch.nn.functional import cross_entropy
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as Loss  # The preceding comment line is to suppress a spurious warning message.

from features import Alphabet


class Dimensions(NamedTuple):

    a: int
    """Number of symbols in the alphabet"""

    b: int
    """Number of morphemes in a batch"""

    c: int
    """Size of a symbol vector"""

    m: int
    """Number of symbols in a morpheme"""


class UnbindingLoss(Loss):
    """
    This criterion generalizes cross-entropy loss
       over predicted symbol vectors in a tensor product representation of a morpheme
       and the corresponding gold standard vectors, using cosine similarity.
    """

    def __init__(
        self,
        alphabet: Alphabet,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__(reduction=reduction)

        self.register_buffer("alpha_tensor", torch.stack([torch.tensor(symbol.vector) for symbol in alphabet]))
        """Tensor containing gold standard vector representations for each symbol in the alphabet"""

        self.ignore_index = alphabet.index_of(alphabet.pad)
        """When calculating the loss, padding should not contribute to the loss. See torch.nn.CrossEntropyLoss"""

        self.weight = weight
        """See torch.nn.CrossEntropyLoss for details"""

        self.a: int = len(alphabet)
        """Number of symbols in the alphabet"""

        self.c: int = len(alphabet.pad.vector)
        """Size of a symbol vector"""

    def check_dimensions(self, predicted: torch.Tensor, label: torch.Tensor) -> Dimensions:

        errors: List[str] = list()

        if len(predicted.shape) != 3:
            errors.append(f"Predicted tensor should have 3 dimensions but actually has {len(predicted.shape)}.")
        if len(label.shape) != 3:
            errors.append(f"Label tensor should have 3 dimensions but actually has {len(label.shape)}.")
        if len(errors) > 0:
            raise ValueError("\n".join(errors))
        if predicted.shape[0] != label.shape[0]:
            errors.append(f"Initial dimension (representing batch size)" +
                          f" of predicted and label tensors must be the same, but is not:" +
                          f" {predicted.shape[0]} != {label.shape[0]}.")
        if predicted.shape[1] != label.shape[1]:
            errors.append(f"Second dimension (representing number of symbols per morpheme)" +
                          f" of predicted and label tensors must be the same, but is not:"
                          f" {predicted.shape[1]} != {label.shape[1]}.")
        if predicted.shape[2] != self.c:
            errors.append(f"Final dimension of predicted tensor must match" +
                          f" expected size of symbol vector, but does not:" +
                          f" {predicted.shape[2]} != {self.c}.")
        if label.shape[2] != self.a:
            errors.append(f"Final dimension of label tensor must match" +
                          f" number of symbols in the alphabet, but does not:" +
                          f" {label.shape[2]} != {self.a}.")
        if len(errors) > 0:
            raise ValueError("\n".join(errors))
        else:
            return Dimensions(a=self.a, b=predicted.shape[0], c=self.c, m=predicted.shape[1])

    def calculate_cosine_similarity(self, predicted: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between predicted symbol vectors and gold standard symbol vectors.

        predicted: Pytorch tensor with shape (b,m,c) representing a batch (b) of morphemes,
                        where each morpheme consists of a sequence of m symbols,
                         and each symbol is represented by a vector of length c
        gold:      Pytorch tensor with shape (a,c) representing the gold standard symbol vectors
                        for each symbol in the alphabet. The alphabet consists of a symbols.

        The meaning of the above dimensions is as follows:
        * a -> the number of symbols in the alphabet
        * b -> batch size; the number of morphemes in the predicted tensor
        * c -> the size of an individual symbol vector
        * m -> the maximum number of symbols in a morpheme

        This function calculates a Pytorch tensor of shape (b,m,a) representing the cosine similarity
            between a predicted character at a given position in a morpheme and all symbols in the alphabet:

            Let pred represent the symbol vector predicted[b][m], with length c
            Let gold represent the symbol vector gold[a], with length c

            Then cosine_similarity[b][m][a] = (pred • gold) / ||pred|| ||gold||
                and this value represents the cosine similarity between
                the predicted m^th symbol vector in the b^th batch and
                the symbol vector for the a^th symbol in the alphabet.
        """

        dimensions = self.check_dimensions(predicted, label)

        # Calculate the dot product between predicted[y][z] and self.alpha_tensor[x]
        #   for each batch y in range(0, b),
        #       each character position z in range(0, m),
        #   and each symbol index x in range(0, a).
        dot_product = torch.einsum("bmc,ac->bma", predicted, self.alpha_tensor)

        # Calculate a tensor of shape (a),
        #   where gold_norm[x] is the Euclidean norm of the x^th symbol vector in the alphabet
        gold_norm = torch.norm(self.alpha_tensor, p=2, dim=-1)

        # Calculate a tensor of shape (b,m),
        #   where pred_norm[y][z] is the Euclidean norm of predicted symbol vector at position z of batch y
        pred_norm = torch.norm(predicted, p=2, dim=-1)

        # Expand gold_norm to have the same shape as dot_product (b,m,a)
        reshaped_gold_norm = gold_norm.unsqueeze(0).unsqueeze(0).expand(dimensions.b, dimensions.m, -1)

        # Expand pred_norm to have the same shape as dot_product (b,m,a)
        reshaped_pred_norm = pred_norm.unsqueeze(-1).expand(-1, -1, dimensions.a)

        # cosine_similarity = (pred • gold) / (||pred|| * ||gold||)
        cosine_similarity = dot_product / (reshaped_gold_norm * reshaped_pred_norm)

        return cosine_similarity  # Shape: (b,m,a)

    def forward(self, predicted, label):

        # Calculate and reshape cosine_similarity into shape (b*m, a)
        cosine_similarity = self.calculate_cosine_similarity(predicted, label).view(-1, self.a)

        # Get the index of the correct symbol for each morpheme position in each batch,
        #   then reshape into shape (b*m)
        gold_labels = label.view(-1, self.a).argmax(dim=1)

        return cross_entropy(
            cosine_similarity,
            gold_labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
        )

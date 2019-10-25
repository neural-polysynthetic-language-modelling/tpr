from typing import List

import torch

from alphabet import Alphabet


class TensorProductRepresentation:

    # @staticmethod
    # def sequence_to_tensor(
    #         sequence: List[str], alphabet: Alphabet, max_sequence_length: int,
    # ) -> torch.Tensor:
    #
    #     character_vector_size: int = len(alphabet[alphabet.oov])
    #
    #     result: torch.Tensor = torch.zeros(character_vector_size, max_sequence_length)
    #
    #     # Process characters in the sequence
    #     for i, char in enumerate(sequence):
    #
    #         if i < max_sequence_length:
    #             # Look up the vector of integers representing the i^th character in the sequence
    #             char_vector: torch.Tensor = torch.tensor(alphabet[char].vector)
    #
    #             # Construct a one-hot role vector for character position i
    #             role_vector: torch.Tensor = torch.zeros(max_sequence_length)
    #             role_vector[i] = 1
    #
    #             # Calculate result[j][k] += char_vector[j] * role_vector[k]
    #             equation_in_einstein_notation: str = "...j,k->...jk"
    #             result += torch.einsum(equation_in_einstein_notation, [char_vector, role_vector])
    #
    #     else:
    #         logging.warning(f"WARNING - Skipping character at index {i} of {''.join(sequence)}" +
    #                         f"because it would exceed maximum length of {max_sequence_length}.")
    #
    #     # Treat anything after the morpheme as being filled by alphabet.padding_symbol
    #     char_vector = torch.tensor(alphabet[alphabet.pad].vector)
    #     for i in range(i + 1, max_sequence_length):
    #
    #         # Construct a one-hot role vector for character position i
    #         role_vector: torch.Tensor = torch.zeros(max_sequence_length)
    #         role_vector[i] = 1
    #
    #         # Calculate result[j][k] += char_vector[j] * role_vector[k]
    #         equation_in_einstein_notation: str = "...j,k->...jk"
    #         result += torch.einsum(equation_in_einstein_notation, [char_vector, role_vector])
    #
    #     return result

    @staticmethod
    def sequence_to_tensor(sequence: List[str], alphabet: Alphabet) -> torch.Tensor:

        length_of_sequence: int = len(sequence)

        result: torch.Tensor = torch.zeros(alphabet.vector_length, length_of_sequence)

        # Process characters in the sequence
        for i, char in enumerate(sequence):

            # Look up the vector of integers representing the i^th character in the sequence
            char_vector: torch.Tensor = torch.tensor(alphabet[char].vector)

            # Construct a one-hot role vector for character position i
            role_vector: torch.Tensor = torch.zeros(length_of_sequence)
            role_vector[i] = 1

            # Calculate result[j][k] += char_vector[j] * role_vector[k]
            equation_in_einstein_notation: str = "...j,k->...jk"
            result += torch.einsum(equation_in_einstein_notation, [char_vector, role_vector])
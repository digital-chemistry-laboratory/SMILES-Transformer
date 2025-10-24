from .basetokenizertemplate import BaseTokenizerTemplate
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
from collections import Counter

import math
import re
import os
import pathlib

SMILES_REGEX = r"\[[^\[\]]+\]|Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|[A-Z][a-z]?|\d|=|#|\/|\\|\(|\)|\.|:|\+|\-|\@"


class BPETokenizer(BaseTokenizerTemplate):
    """
    BPE Tokenizer for SMILES strings
    """

    tokenizer_kind = "bpe"

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        num_workers=None,
        **kwargs,
    ):
        vocab_file_path = pathlib.Path(vocab_file)
        self.vocab_path = vocab_file_path
        self.merges_path = vocab_file_path.parent / "merges_BPE.txt"

        super().__init__(self.vocab_path, **kwargs)
        # Default special tokens
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        # Initialize vocabulary and merges
        if merges_file:
            self.merges_path = pathlib.Path(merges_file)
        else:
            self.merges_path = self.vocab_path.parent / "merges_BPE.txt"

        # Attempt to load BPE vocab and merges if both files exist
        if self.vocab_path.exists() and self.merges_path.exists():
            self._load()
        else:
            self.token_to_id = {}
            self.id_to_token = {}
            self.merges = []

    @staticmethod
    def _initial_tokenize_worker(text: str) -> List[str]:
        """Helper to tokenize a single SMILES string."""
        return list(text)

    @staticmethod
    def _get_pairs_chunk(sequence_chunk: List[List[str]]) -> Counter:
        """Helper to count pairs in a chunk of sequences."""
        pairs = Counter()
        for sequence in sequence_chunk:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pairs[pair] += 1
        return pairs

    @staticmethod
    def _apply_merge_chunk(
        args: Tuple[List[List[str]], Tuple[str, str]],
    ) -> List[List[str]]:
        """Helper to apply a merge to a chunk of sequences."""
        sequence_chunk, pair_to_merge = args
        new_sequences_chunk = []
        merged_token = pair_to_merge[0] + pair_to_merge[1]

        for sequence in sequence_chunk:
            i = 0
            new_sequence = []
            while i < len(sequence):
                if (
                    i < len(sequence) - 1
                    and sequence[i] == pair_to_merge[0]
                    and sequence[i + 1] == pair_to_merge[1]
                ):
                    new_sequence.append(merged_token)
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            new_sequences_chunk.append(new_sequence)
        return new_sequences_chunk

    def _initial_tokenize(self, text: str) -> List[str]:
        """
        Initial tokenization of SMILES string by splitting into characters
        """
        return list(text)

    def _get_pairs(
        self, sequences: List[List[str]], executor: ProcessPoolExecutor
    ) -> Counter:
        """
        Count frequency of adjacent token pairs
        """
        pairs = Counter()
        num_active_workers = min(self.num_workers, len(sequences))
        chunk_size = math.ceil(len(sequences) / num_active_workers)
        chunks = [
            sequences[i : i + chunk_size]
            for i in range(0, len(sequences), chunk_size)
            if sequences[i : i + chunk_size]
        ]
        chunk_counters = list(executor.map(BPETokenizer._get_pairs_chunk, chunks))

        total_pairs = Counter()
        for counter_result in chunk_counters:
            total_pairs.update(counter_result)

        return total_pairs

    def _apply_merge(
        self,
        pair: Tuple[str, str],
        sequences: List[List[str]],
        executor: ProcessPoolExecutor,
    ) -> List[List[str]]:
        """
        Apply a BPE merge to all sequences by replacing every occurrence of the specified token pair with a new merged token.
        """
        if not sequences:
            return []

        num_active_workers = min(self.num_workers, len(sequences))
        if num_active_workers == 0:
            return []

        chunk_size = math.ceil(len(sequences) / num_active_workers)

        tasks = []
        for i in range(0, len(sequences), chunk_size):
            chunk = sequences[i : i + chunk_size]
            if chunk:
                tasks.append((chunk, pair))

        if not tasks:
            return []

        new_sequences = []

        processed_chunks = list(executor.map(self._apply_merge_chunk, tasks))

        for chunk_result in processed_chunks:
            new_sequences.extend(chunk_result)

        return new_sequences

    def train(
        self,
        smiles_list: List[str],
        vocab_size: int = 500,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """
        Train BPE tokenizer directly on a list of SMILES strings.
        """
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            sequences = self._tokenize_smiles(smiles_list, executor)
            next_id = self._build_initial_vocabulary(sequences)
            self._run_bpe_training(
                sequences, vocab_size, min_frequency, show_progress, next_id, executor
            )
        self.save()

    def _tokenize_smiles(
        self, smiles_list: List[str], executor: ProcessPoolExecutor
    ) -> List[List[str]]:
        """
        Tokenize each SMILES string using the _initial_tokenize method.
        """
        return list(executor.map(self._initial_tokenize_worker, smiles_list))

    def _build_initial_vocabulary(self, sequences: List[List[str]]) -> int:
        """
        Build the initial vocabulary from tokenized sequences.
        """
        # Create initial vocabulary from all unique tokens
        all_tokens = set()
        for sequence in sequences:
            all_tokens.update(sequence)

        # Initialize vocabulary with special tokens
        self.token_to_id = {token: i for i, token in enumerate(self.all_special_tokens)}
        next_id = len(self.all_special_tokens)

        # Add initial tokens to vocabulary
        for token in sorted(all_tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                next_id += 1

        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        return next_id

    def _run_bpe_training(
        self,
        sequences: List[List[str]],
        vocab_size: int,
        min_frequency: int,
        show_progress: bool,
        next_id: int,
        executor: ProcessPoolExecutor,
    ) -> None:
        """
        Perform the BPE training loop until the vocabulary size reaches the desired size.
        """
        num_merges = vocab_size - len(self.token_to_id)
        self.merges = []
        last_percent = -1
        print("\nvocab_size:", vocab_size)

        for i in range(num_merges):
            # Show progress
            if show_progress:
                if i == 0:
                    print("Training BPE tokenizer...")
                percent = int((i + 1) / num_merges * 100)
                if percent % 10 == 0 and percent != last_percent:
                    print(f"Progress: {percent}%")
                    last_percent = percent

            # Find most frequent pair
            pair_counts = self._get_pairs(sequences, executor)

            # Filter by minimum frequency
            filtered_pairs = [
                (pair, count)
                for pair, count in pair_counts.items()
                if count >= min_frequency
            ]

            if not filtered_pairs:
                break

            # Get the most frequent pair
            best_pair, best_count = max(filtered_pairs, key=lambda x: x[1])

            # Add the merged token to vocabulary if it doesn't already exist
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = next_id
                self.id_to_token[next_id] = merged_token
                next_id += 1

            # Merge the pair in all sequences
            sequences = self._apply_merge(best_pair, sequences, executor)

            # Record the merge operation
            self.merges.append(best_pair)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a SMILES string into subword tokens
        """
        # Start with initial tokens
        tokens = self._initial_tokenize(text)

        # Apply merges in order
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = tokens[i] + tokens[i + 1]
                    tokens.pop(i + 1)
                else:
                    i += 1

        return tokens

    def _load(self) -> None:
        self.token_to_id = {}
        with open(self.vocab_path, "r") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.token_to_id[token] = idx

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.merges = []
        with open(self.merges_path, "r") as f:
            for line in f:
                pair = tuple(line.strip().split())
                self.merges.append(pair)

    def encode(self, text: str) -> List[int]:
        """
        Encode a SMILES string into token IDs
        """
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                sub_tokens = re.findall(SMILES_REGEX, token)
                if sub_tokens:
                    for sub in sub_tokens:
                        if sub in self.token_to_id:
                            ids.append(self.token_to_id[sub])
                        else:
                            for char in sub:
                                if char in self.token_to_id:
                                    ids.append(self.token_to_id[char])
                                else:
                                    ids.append(self.token_to_id[self.unk_token])
                else:
                    for char in token:
                        if char in self.token_to_id:
                            ids.append(self.token_to_id[char])
                        else:
                            ids.append(self.token_to_id[self.unk_token])
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to a SMILES string
        """
        tokens = []
        unk_count = 0
        unk_id = self.token_to_id[self.unk_token]

        for id_ in ids:
            token = self.id_to_token.get(id_, self.unk_token)
            tokens.append(token)
            if id_ == unk_id:
                unk_count += 1

        smiles = "".join(tokens)
        # Remove special tokens inserted during encoding, but keep [UNK] tokens
        for special_token in self.all_special_tokens:
            if special_token is not self.unk_token:
                smiles = smiles.replace(special_token, "")

        return smiles, unk_count

    def save(self) -> Tuple[str, str]:
        """
        Save tokenizer vocabulary and merge rules
        """
        # os.makedirs(save_directory, exist_ok=True)

        # Save vocabulary
        with open(self.vocab_path, "w") as f:
            for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
                f.write(token + "\n")

        # Save merges

        with open(self.merges_path, "w") as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size
        """
        return len(self.token_to_id)


import cProfile, pstats

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    import pandas as pd

    csv_path = r"/Users/stelios/Library/CloudStorage/OneDrive-epfl.ch/1. ETH/Semester 2/z. PB/smiles-cgr-transformers/data/interim/phosphatase.csv"
    df = pd.read_csv(csv_path).head(10000)
    smiles_list = df["AAM"].tolist()

    tokenizer = BPETokenizer()

    from timeit import default_timer as timer

    start = timer()
    tokenizer.train(
        smiles_list=smiles_list, vocab_size=200, min_frequency=2, show_progress=True
    )
    end = timer()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)
    print("time elapsed:", end - start)
    tokenizer.save(".")

    test_smiles = "[NH$2:1$][$CH2:2][C:3](=$[O:4])[OH:5]"
    tokens = tokenizer.tokenize(test_smiles)
    token_ids = tokenizer.encode(test_smiles)
    reconstructed, unk_count = tokenizer.decode(token_ids)

    # Check reuslts
    print(f"Original SMILES: {test_smiles}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Unknown tokens: {unk_count}")

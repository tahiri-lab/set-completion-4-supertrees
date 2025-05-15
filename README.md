# Phylogenetic tree set completion in the context of supertrees

This project introduces an innovative method for completing a set of phylogenetic trees by minimizing the addition of new taxa while maintaining key evolutionary signals. Building on existing tree completion techniques, it incorporates consensus maximal completion subtrees and combined branch length data to enhance the process. This ensures that the completion is both biologically relevant and computationally efficient. Each tree in the set is iteratively completed by adding missing taxa using consensus subtrees derived from overlapping information in the other trees. The outcome is a collection of completed trees that serve as a foundation for creating multiple comprehensive consensus trees, which provide a cluster-based depiction of alternative supertrees.

>**Please note that this project is under development and is not a final version.** Features and functionalities are subject to change, and the tool may undergo significant updates. Use it at your own discretion, and feel free to contribute or provide feedback to help improve its capabilities.

## Features

- **Preserves Evolutionary Signals**: Maintains key evolutionary relationships and branch lengths.
- **Consensus-Based Approach**: Utilizes consensus maximal completion subtrees and aggregated branch lengths.
- **Biologically Meaningful**: Ensures the completion process is both relevant and computationally optimized.
- **Comprehensive Output**: Generates completed trees as intermediates for constructing multiple consensus supertrees.

## Usage

### Input Format

- **Input folder**: Place your input phylogenetic tree multisets in the `input_multisets` directory.
- **File naming**: Each multiset should be in a separate text file named `multiset_X.txt`, where `X` is a unique identifier (e.g., `multiset_1.txt`).
- **Tree format**: Each line in the input file should contain a single phylogenetic tree in Newick format.


### Running the Script

The main script is `multiset_completion.py`. To execute the script:

1. **Ensure input files are in place**

   Place all your multiset files in the `input_multisets` directory.

2. **Run the script**

   ```bash
   python multiset_completion.py.py
   ```

   *Note: The script is configured to process files named `multiset_X.txt` in the `input_multisets` folder and output completed trees to the `completed_multisets` folder as `completed_multiset_X.txt`.*

### Output Format

- **Output folder**: Completed phylogenetic tree multisets are saved in the `completed_multisets` directory.
- **File naming**: Each completed multiset file corresponds to its input, named `completed_multiset_X.txt`.
- **Tree format**: Each line in the output file contains a completed phylogenetic tree in Newick format.


## Dependencies

- **Python 3.6 or higher**
- **ETE3**: For phylogenetic tree manipulation.
- **NumPy**: For numerical computations.

Install dependencies using:

```bash
pip install ete3 numpy
```

*Additional dependencies may be required based on future updates.*


## License

This project is licensed under the [MIT License](LICENSE).

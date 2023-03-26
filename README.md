# Simple Linear Iterative Clustering (SLIC)

This repo implements [SLIC](https://infoscience.epfl.ch/record/149300) with numpy from scratch.

## Requirements

Install requirements:

```bash
pip3 install -r requirements.txt
```

Code tested under Python 3.10, numpy 1.23.5 and Pillow 9.4.0

## Run

```bash
python3 main.py [-h] [-i INPUT_PATH] -o OUTPUT_PATH -k NUM_OF_CENTROIDS [--max-iter MAX_ITER] [-t CONVERGE_THRESHOLD]
```

where 

```
options:
  
  -i, --input-path         path to the input
  -o, --output-path        path to the output
  -k, --num-of-centroids   number of centroids
  --max-iter               maximum number of iterations
  -t, --converge-threshold if centroids move less than this value after certain iteration, then stop iterating
```

## Example

This example uses 59 centroids and 10 iterations

<img src="doc/img.jpg" width="48%"> <img src="doc/output.png" width="48%">

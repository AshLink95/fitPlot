# fitPlot
A simple lab 1-stop-shop utility for all plot fitting purposes.

## Installation
In the cloned repository,
```bash
# Create and activate virtual environment
python -m venv .
source bin/activate  # On Windows: bin\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Use
It supports plotting the following predefined functions:
* Lorentzian
* Gaussian
* Exponential
* Exponential Decay
* Power law
* Linear
* Quadratic
* Cubic
* Sinusoidal
* Damped oscillation
* Logarithmic
* Stretched Exponential

As well as the ability to input custom functions.

You can input all information in the cli or simply use a csv file.

To start fitting,
```bash
source bin/activate  # On Windows: bin\Scripts\activate
python fitPlot.py
```
and follow the CLI to fit your needs.

You can fit multiple datasets, each with its unique fitting function with automatic estimation and statistical output.

## Test
1 test file was provided. It contains 3 datasets of 50 points, each. These datasets are fitted by the logarithmic, exponantial decay and quadratic options.

This test is simple but showcases the capability of this utility.

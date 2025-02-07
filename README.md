# Python "Neural Networks and Deep Learning"

This repository provides functions to train a simple AI using backpropagation, including snippets for extracting training and validation data. The code has been enhanced with **NumPy** and **PyTensor**.

## Updates
- The code has been updated to be compatible with **Python 3.11**.
- Libraries have been enhanced with the use of **NumPy** and **PyTensor**.
- **PyTensor version 2.27.1** is utilized.

## Installation

### Setting up a Virtual Environment

To isolate your project environment, use `virtualenv`. Here's how you can set it up:

1. **Install virtualenv** if you haven't already:
```sh
pip install virtualenv
```
2. **Create a virtual environment** in the project directory:
```sh
python -m virtualenv .venv
```
3. **Activate the virtual environment**:
- On Windows:
```sh
.venv\Scripts\activate
```
On macOS and Linux:
```sh
source .venv/bin/activate
```
Once activated, your command prompt should change to indicate the virtual environment is active.

### Requirements
Before you can run the code, you need to install the following dependencies:

- Python 3.11
- NumPy (latest compatible version)
- PyTensor 2.27.1

You can install these dependencies using the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Usage
The code in this repository:

- Provides functions to train a simple AI using backpropagation.
- Includes snippets to extract training data and validation data.
- No new features will be added to this repository.
- Bug reports are welcome. Please submit them via the issue tracker.
- Feel free to fork and modify the code for your own purposes.

Notes
- This repository will not be updated for compatibility with future Python versions or library updates beyond what's currently specified.
- Enhancements have been made using NumPy for numerical operations and PyTensor for symbolic computation.
- Make sure your `requirements.txt` file lists `numpy` and `pytensor` with their respective versions or constraints.
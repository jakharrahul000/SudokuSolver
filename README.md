# SudokuSolver
This is a machine learning project to solve Sudoku puzzle from images.

The model was trained using Keras and has been saved in the num_detector.model and saved_model.pb.

First the sudoku puzzle is extracted from the images provided, then it is rotated to make it vertical.
After this, all the cells are extracted individually.
These cells are then passed to the model which give the number it is representing.

backtracking is used to solve the extracted sudoku puzzle.

To run the code, run solvesudoku.py file.

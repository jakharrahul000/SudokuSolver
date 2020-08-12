from extractsudokupuzzle import ExtractSudokuPuzzle
from extractsudokucell import ExtractSudokuCell
from digitrecognition import sudoku

import cv2
import matplotlib.pyplot as plt

def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end=' ')
        print()

def find_empty_location(arr,l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]==0):
                l[0]=row
                l[1]=col
                return True
    return False

def used_in_row(arr,row,num):
    for i in range(9):
        if(arr[row][i] == num):
            return True
    return False

def used_in_col(arr,col,num):
    for i in range(9):
        if(arr[i][col] == num):
            return True
    return False

def used_in_box(arr,row,col,num):
    for i in range(3):
        for j in range(3):
            if(arr[i+row][j+col] == num):
                return True
    return False

def check_location_is_safe(arr, row, col, num):
    return not used_in_row(arr, row, num) and not used_in_col(arr, col, num) and not used_in_box(arr, row - row % 3, col - col % 3, num)

def solve_sudoku(arr):
    l = [0, 0]
    if (not find_empty_location(arr, l)):
        return True
    row = l[0]
    col = l[1]
    for num in range(1, 10):
        if (check_location_is_safe(arr, row, col, num)):
            arr[row][col] = num
            if (solve_sudoku(arr)):
                return True
            arr[row][col] = 0
    return False

def SolveSudoku(img_to_be_processed):
  # img_to_be_processed = cv2.imread('../img/hard.png', cv2.IMREAD_GRAYSCALE)
    
  postProcessedExtracted = ExtractSudokuPuzzle(img_to_be_processed)

  cellImagesbyFeatures = ExtractSudokuCell(postProcessedExtracted, check=True)
  grid = sudoku(cellImagesbyFeatures)

  if solve_sudoku(grid):
    return grid
  
  cellImagesbyContours = ExtractSudokuCell(postProcessedExtracted, check=False)
  grid = sudoku(cellImagesbyContours)

  if solve_sudoku(grid):
    return grid
  
  return None
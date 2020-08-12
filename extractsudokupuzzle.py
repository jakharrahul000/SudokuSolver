import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
from copy import deepcopy
from skimage import exposure
import matplotlib.pyplot as plt

def preprocessImage(sudokuImage):
  # blurr filter
  blurred=cv2.bilateralFilter(sudokuImage,5,75,75)

  #smoothen contour
  kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
  closed=cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

  #normalization
  div=np.float32(blurred)/(closed)
  normalized=np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))

  # apply inv threshold to convert into binary
  threshold=cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

  if((threshold==0).all()):
    threshold=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

  return threshold

def findLargestContour(processed):
  contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # find largest ContourArea
  maxArea, minArea = 0, 300
  cont=None
  for cnt in contours:
    cntArea=cv2.contourArea(cnt)
    if cntArea>minArea:
      #find perimeter
      cntPerimeter=0.02*cv2.arcLength(cnt, True)
      # approx to cnt
      polyApprox=cv2.approxPolyDP(cnt, cntPerimeter, True)

      if cntArea>maxArea and len(polyApprox)==4:
        cont=polyApprox
        maxArea=cntArea

  return cont, maxArea

def computeBoundingBoxOfFeature(processed, seed, boundingBox=True):
  img=deepcopy(processed)

  height, width=processed.shape[:2]
  mask=np.zeros((height+2, width+2), np.uint8)

  # floodFill all features with gray
  for y in range(height):
    for x in range(width):
      if img.item(y,x)==255 and x<width and y<height:
        cv2.floodFill(img, None, (x,y), 64)
  
  # fill connected component with seed with white
  if seed is not None:
    if seed[0] is not None and seed[1] is not None:
      cv2.floodFill(img, mask, seed, 255)
  
  topLine, bottomLine, leftLine, rightLine = height, 0, width, 0
  topLeft, topRight, bottomLeft, bottomRight = (width,height), (0,height), (width,0), (0,0)

  for y in range(height):
    for x in range(width):
      # color every non-target feature with black
      if img.item(y,x)==64:
        cv2.floodFill(img, mask, (x,y), 0)
      # compute bounding box of target feature
      if img.item(y,x)==255:
        if boundingBox:
          if x<leftLine: leftLine=x
          if x>rightLine: rightLine=x
          if y<topLine: topLine=y
          if y>bottomLine: bottomLine=y
        else:
          if x+y<sum(topLeft): topLeft=(x,y)
          if x+y>sum(bottomRight): bottomRight=(x,y)
          if x-y>topRight[0]-topRight[1]: topRight=(x,y)
          if x-y<bottomLeft[0]-bottomLeft[1]: bottomLeft=(x,y)
  
  if boundingBox:
    topLeft=(leftLine, topLine)
    bottomRight=(rightLine, bottomLine)
    cornerPoints=np.array([topLeft, bottomRight], dtype='float32')
  else:
    cornerPoints=np.array([topLeft, topRight, bottomRight, bottomLeft], dtype='float32')
  
  return img, cornerPoints

def findLargestFeature(processed, topLeft=None, bottomRight=None):
  preprocessed=deepcopy(processed)

  height, width=processed.shape[:2]
  if topLeft is None:
    topLeft=(0,0)
  if bottomRight is None:
    bottomRight=(width, height)
  
  if bottomRight[1]-topLeft[1]>height or bottomRight[0]-topLeft[0]>width:
    raise ValueError("Error in findLargestFeature: coordinate are out of bound")

  maxArea=0
  seed=None

  for y in range(topLeft[1], bottomRight[1]):
    for x in range(topLeft[0], bottomRight[0]):
      if preprocessed.item(y,x)==255 and x<width and y<height:
        #floodFill to find largest feature
        featureArea=cv2.floodFill(preprocessed, None, (x,y), 64)
        if featureArea[0]>maxArea:
          maxArea=featureArea[0]
          seed=(x,y)
  
  feature, cornerPoints = computeBoundingBoxOfFeature(processed, seed, boundingBox=False)
  return feature, cornerPoints, seed

def getQuadrangleVertices(grid):
  if len(grid)==0: return None

  corners=grid.reshape(len(grid), 2)
  quad=np.zeros((4,2), dtype='float32')

  s=corners.sum(axis=1)
  #topLeft and bottomRight will have min and max sum
  quad[0]=corners[np.argmin(s)]
  quad[2]=corners[np.argmax(s)]

  d=np.diff(corners, axis=1)
  #topRight and bottomLeft will have min and max difference
  quad[1]=corners[np.argmin(d)]
  quad[3]=corners[np.argmax(d)]

  return quad

def findSudokuPuzzleGrid(processed, original):
  height, width=processed.shape[:2]
  processedArea=height*width

  # Find sudoku puzzle via largest contour
  largestContour, largestContourArea=findLargestContour(processed)

  # Find sudoku puzzle via largest feature
  feature, cornerPoints, seed=findLargestFeature(processed)

  featureCornerPoints=cornerPoints.astype(int)
  featureCornerPoints=featureCornerPoints.tolist()
  topLeft,topRight,bottomRight,bottomLeft=featureCornerPoints
  topLeft, topRight, bottomRight, bottomLeft = tuple(topLeft), tuple(topRight), tuple(bottomRight), tuple(bottomLeft)

  largestFeatureArea=cv2.contourArea(cornerPoints)
  
  try:
    ratio=largestFeatureArea/largestContourArea
  except(ZeroDivisionError):
    if largestFeatureArea==0:
      print("Error in findSudokuPuzzleGrid: Unable to extract puzzle from image.")
      exit()
    else:
      ratio=0
  
  if ratio<0.95 or ratio>1.5:
    # Use largest feature
    cv2.line(original,topLeft,topRight,(0,255,0),3)
    cv2.line(original,topLeft,bottomLeft,(0,255,0),3)
    cv2.line(original,topRight,bottomRight,(0,255,0),3)
    cv2.line(original,bottomLeft,bottomRight,(0,255,0),3)
    return cornerPoints
  else:
    #use largest contour
    cv2.drawContours(original,[largestContour],-1,(0,255,0),3)    
    return getQuadrangleVertices(largestContour)

def computeMaxWidthAndHeightOfSudokuPuzzle(quad):
	topLeft,topRight,bottomRight,bottomLeft=quad
	
	upperWidth=np.sqrt( ((topRight[0]-topLeft[0])**2) + ((topRight[1]-topLeft[1])**2) )
	bottomWidth=np.sqrt( ((bottomRight[0]-bottomLeft[0])**2) + ((bottomRight[1]-bottomLeft[1])**2) )

	leftHeight=np.sqrt( ((topLeft[0]-bottomLeft[0])**2) + ((topLeft[1]-bottomLeft[1])**2) )
	rightHeight=np.sqrt( ((topRight[0]-bottomRight[0])**2) + ((topRight[1]-bottomRight[1])**2) )

	maxWidth=max(int(upperWidth),int(bottomWidth))
	maxHeight=max(int(leftHeight),int(rightHeight))
	return maxWidth,maxHeight

def extractSudokuPuzzleAndWarpPerspective(quad,maxWidth,maxHeight,original):
	original=deepcopy(original)
	
	#bird eye view
	dstPoints=np.array([ [0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1] ],dtype="float32")
	#compute perspective transform
	M=cv2.getPerspectiveTransform(quad,dstPoints)
	#apply transformation
	warp=cv2.warpPerspective(original,M,(maxWidth,maxHeight))
	
	return warp

def postProcessExtractedSudokuPuzzle(warpedSudokuPuzzle):
	postProcessed=exposure.rescale_intensity(warpedSudokuPuzzle,out_range=(0,255))
	postProcessed = cv2.resize(postProcessed,(450, 450),interpolation=cv2.INTER_AREA)
	return postProcessed

def ExtractSudokuPuzzle(img_to_be_processed):
  original = img_to_be_processed

  processed = preprocessImage(original)

  solid_grid_puzzle = findSudokuPuzzleGrid(processed, original)

  maxWidth,maxHeight=computeMaxWidthAndHeightOfSudokuPuzzle(solid_grid_puzzle)

  warpedSudokuPuzzle= extractSudokuPuzzleAndWarpPerspective(solid_grid_puzzle,maxWidth,maxHeight,original)

  postProcessedExtracted=postProcessExtractedSudokuPuzzle(warpedSudokuPuzzle)

  return postProcessedExtracted



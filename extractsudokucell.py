import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def preProcessCellImage(cellImage):
  denoised=cv2.fastNlMeansDenoising(src=cellImage, h=10, templateWindowSize=9, searchWindowSize=13)
  # blurr image
  blurred=cv2.bilateralFilter(denoised, d=15, sigmaColor=40, sigmaSpace=40)
  # apply inverse threshold
  threshold=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 3)
  # remove circle, etc
  kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
  dilate=cv2.dilate(threshold, kernel, iterations=1)
  erode=cv2.erode(dilate, kernel, iterations=1)

  return erode

def sudokuCellPositions(postProcessedExtracted):
  cellPositions=[]
  height, width = postProcessedExtracted.shape[:2]
  cellHeight, cellWidth = height//9, width//9
  x1, y1, x2, y2 = 0, 0, 0, 0

  for i in range(9):
    y2=y1+cellHeight
    x1=0
    for j in range(9):
      x2=x1+cellWidth
      cellPositions.append([x1,x2,y1,y2])
      x1=x2
    y1=y2
  
  return cellPositions

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

def cellImageByLargestFeature(cellImage, re_extract=False):
  height, width=cellImage.shape[:2]
  index=int(np.mean([height,width])/2.5)

  if(not re_extract):
    _, boundingBox, seed=findLargestFeature(cellImage, [index-2,index-2], [width-index,height-index])
  else:
    _, boundingBox, seed=findLargestFeature(cellImage, [index,index], [width-index,height-index])
  
  feature, cornerPoints=computeBoundingBoxOfFeature(cellImage, seed, boundingBox=True)
  topLeft, bottomRight=cornerPoints
  x1,x2,y1,y2=topLeft[0],bottomRight[0],topLeft[1],bottomRight[1]
  width, height=x2-x1, y2-y1

  if width<0 and height<0 and not re_extract:
    height, width=cellImage.shape[:2]
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    dilate=cv2.dilate(cellImage[6:height, 6:width], kernel, iterations=1)
    return cellImageByLargestFeature(dilate, re_extract=True)
  
  if( (width>0 and height>0 and (width*height)>100) or (width>1 and height>13 and (width*height)>45 and not re_extract) or 
																(width>2 and height>17 and (width*height)>75 and re_extract) ):
    feature=feature[int(y1):int(y2), int(x1):int(x2)]
    return feature
  else:
    return None

def cellImageByLargestContour(cellImage, re_extract=False):
  contours, hierarchy=cv2.findContours(cellImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  maxArea=0
  cx,cy,height,width=None,None,None,None

  for contour in contours:
    x,y,w,h=cv2.boundingRect(contour)
    area=w*h

    if (x>5 and y>1 and area>maxArea and area>65 and ((True) if not re_extract else (x<38 and y<38)) ):
      maxArea=area
      cx,cy,width,height=x,y,w,h
  
  if maxArea>0 and not re_extract:
    if(cx+width>=49 or cy+height>=49):
      height, width=cellImage.shape[:2]
      return cellImageByLargestContour(cellImage[6:height,6:width],re_extract=True)

  if maxArea==0:
    return None
  else:
    return cellImage[cy:cy+height, cx:cx+width]

def centerAndResizeDigit(cellImage, size, padding):
  if(size%2!=0):
    raise ValueError("Error in centerAndResizeDigit: Argument size must be even.")
  if(2*padding>=size):
    raise ValueError("Error in centerAndResizeDigit: Padding cannot be larger than size")
  
  def normalizePadding(length, target):
    if length%2==0:
      p1=p2=(size-length)//2
    else:
      padding=(size-length)//2
      if(length+padding*2+1>target):
        p1=p2=padding
      else:
        p1=padding
        p2=p1+1
    return p1,p2

  height, width=cellImage.shape[:2]
  if width>height:
    leftpadding, rightpadding=padding, padding
    ratio=(size-2*padding)/width
    width, height=int(ratio*width), int(ratio*height)
    if height==0:
      height=1
    cellImage=cv2.resize(cellImage, (width, height))
    topPadding, bottomPadding=normalizePadding(height, width+2*padding)
  else:
    topPadding, bottomPadding=padding, padding
    ratio=(size-2*padding)/height
    width, height=int(ratio*width), int(ratio*height)
    if width==0:
      width=1
    cellImage=cv2.resize(cellImage, (width, height))
    leftPadding, rightPadding=normalizePadding(width, height+2*padding)
  
  cellImage=cv2.copyMakeBorder(cellImage, topPadding, bottomPadding, leftPadding, rightPadding, cv2.BORDER_CONSTANT, None, 0)
  cellImage=cv2.resize(cellImage,(size,size))
  return cellImage

def ExtractSudokuCell(postProcessedExtracted, check=True):
  cellPositions=sudokuCellPositions(postProcessedExtracted)
  blank=np.zeros((28,28), np.uint8)

  digits=[]

  for cell in cellPositions:
    x1,x2,y1,y2 = cell
    cellImage=postProcessedExtracted[y1:y2, x1:x2]

    #Apply image preprocessing to cellImage
    cellImage=preProcessCellImage(cellImage)

    if check:
      # find cellImage via largest feature
      cellImage=cellImageByLargestFeature(cellImage)
    else:
      # find cellImage via largest contour
      cellImage=cellImageByLargestContour(cellImage)
    
    if cellImage is not None:
      cellImage=centerAndResizeDigit(cellImage, 28, 2)
      digits.append(cellImage)
    else:
      digits.append(blank)
  
  return digits




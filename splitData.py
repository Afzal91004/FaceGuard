# splitData.py

import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# Remove the output directory if it exists and recreate it
if os.path.exists(outputFolderPath):
    shutil.rmtree(outputFolderPath)
os.makedirs(outputFolderPath)

# Create necessary subdirectories
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# List all files in the input folder
try:
    listNames = os.listdir(inputFolderPath)
except FileNotFoundError:
    raise FileNotFoundError(f"The directory {inputFolderPath} does not exist.")

# Extract unique image names (without file extension)
uniqueNames = list(set(name.split('.')[0] for name in listNames))

# Shuffle image names
random.shuffle(uniqueNames)

# Calculate number of images for each split
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# Adjust train length to account for rounding issues
if lenData != (lenTrain + lenVal + lenTest):
    lenTrain += (lenData - (lenTrain + lenVal + lenTest))

# Split the unique names into train, val, and test sets
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images: {lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# Copy files to respective folders
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        try:
            shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
            shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')
        except FileNotFoundError:
            print(f"File not found: {fileName}")

print("Split Process Completed...")

# Create data.yaml file
dataYaml = f'''path: ../Data
train: ../train/images
val: ../val/images
test: ../test/images

nc: {len(classes)}
names: {classes}'''

with open(f"{outputFolderPath}/data.yaml", 'w') as f:
    f.write(dataYaml)

print("Data.yaml file Created...")

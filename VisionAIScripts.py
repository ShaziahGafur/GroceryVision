import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

#Perform VisionAI Image Recognition on Image BEFORE

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_nameB = os.path.abspath('images/vegetables_before.jpg')

# Loads the image into memory
with io.open(file_nameB, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
itemsBefore = response.label_annotations

#Repeat process for Image AFTER
# The name of the image file to annotate
file_nameA = os.path.abspath('images/vegetables_after.jpg')

# Loads the image into memory
with io.open(file_nameA, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
itemsAfter = response.label_annotations

missingItems = [item for item in itemsBefore if item not in itemsAfter]
newItems = [item for item in itemsAfter if item not in itemsBefore]

print('Labels Before:')
for label in itemsBefore:
    print(label.description)

print('\n')
print('Labels After:')
for label in itemsAfter:
    print(label.description)

print('\n')

print('Items Missing:')
for label in missingItems:
    print(label.description)

print('\n')
print('Items Newly Added:')
for label in newItems:
    print(label.description)
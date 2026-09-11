# Connects to AWS
  #    ↓
# Finds the RGB image folder
  #    ↓
# Selects five images
    #  ↓
# Reads them temporarily
     # ↓
# Displays them in VSCode

import io # allows Python to treat downloaded bytes as an image file
import boto3 # allows Python to interact with AWS services
import matplotlib.pyplot as plt # allows Python to display images

from PIL import Image # opens and convert image files 
from botocore import UNSIGNED # allows access to the public dataset without an AWS account
from botocore.client import Config # configures the AWS connection

# identify the dataset location 
bucket_name = "pohang-canal-dataset"  # variable used to set the datasets cloud storage name
image_folder = "pohang03/stereo/left_images/" # desired data directory in the dataset

# connect to AWS
s3 = boto3.client(
    "s3", # tells boto3 that we are accessing cloud storage
    region_name="us-west-2", # is the AWS region containing the dataset
    config=Config(signature_version=UNSIGNED) # means that no private AWS login crededentials are required
)

# # Find image files in the selected folder
# # this asks AWS to return a list of all files in the specified folder of the dataset
# # results are stored - response
# response = s3.list_objects_v2(
#     Bucket=bucket_name,
#     Prefix=image_folder
# )

# Check the folders inside pohang03
response = s3.list_objects_v2(
    Bucket=bucket_name,
    Prefix="pohang05/",
    Delimiter="/"
)

print("Top-level folders in the dataset:")

for folder in response.get("CommonPrefixes", []):
    print(folder["Prefix"])

# Selects 5 images 
# Looks inside ASW "response" 
# keeps only 5 files endding in .png
# stores their full cloud paths
# [:5] means that only the first 5 files are kept
image_keys = [
    item["Key"]
    for item in response.get("Contents", [])
    if item["Key"].lower().endswith(".png")
][:5]

# This prints the five selected filenames in the terminal
print("Images found:")
for key in image_keys:
    print(key)

# Display the five images
# creates a figure with 1 row and 5 columns, and sets the size of the figure
fig, axes = plt.subplots(1, len(image_keys), figsize=(20, 5))

# This loops through the five 
# axis is the display area for each image
# zip() combines the two lists into pairs of (axis, image_key)
for axis, image_key in zip(axes, image_keys):
    # This downloads the image from AWS
    response = s3.get_object(
        Bucket=bucket_name,
        Key=image_key
    )

    image_data = response["Body"].read() # asks AWS to return one specific image
    image = Image.open(io.BytesIO(image_data)).convert("RGB") # reads the image data into python memory
    #This:
#1. Treats the downloaded data as an image file.
#2. Opens the image using Pillow.
#3. Converts it into standard RGB format.
# The image is not permanently saved to your laptop.

# Displays each image
    axis.imshow(image) # Places the image inside its display area.
    axis.set_title(image_key.split("/")[-1]) # Displays only the image filename as the title. (eg 000001.png)
    axis.axis("off") # Removes the axes and numbers around the image so the image is easier to view.

# Shows the completed figure
plt.tight_layout() # spaces the image neatly 
plt.show() # opens the visual display
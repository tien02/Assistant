# author: https://github.com/tien02
import os
import cv2 as cv
from tqdm import tqdm

# Trich xuat ROI
def extract_roi(img, offsets):
  x1 = int(offsets[0])
  y1 = int(offsets[1])
  x2 = int(offsets[2])
  y2 = int(offsets[3])
  x3 = int(offsets[4])
  y3 = int(offsets[5])
  x4 = int(offsets[6])
  y4 = int(offsets[7])

  top_y = min(y1, y2) 
  bottom_y = max(y3, y4)
  left_x = min(x1, x4)
  right_x = max(x2,x3)

  roi = img[top_y : bottom_y, left_x : right_x, :]
  
  return roi

'''
Chuyen du lieu tu format cua Vin Text sang format cua VietOCR
# Parameter
base_dir: vi tri luu vao
label_dir: vi tri Label
start_index_img: ten anh dat theo so duoc bat dau tu
name: ten folder
annot_name: ten annotation
# Return 
folder img: chua anh crop tu dataset
train_annotation.txt: chua du lieu theo format VietOCR
'''
def generate_data_to_vietocr(base_dir,
                             img_dir,
                             label_dir, 
                             start_index_img, 
                             name = "train_vietocr_format", 
                             annot_name = "train_annotation"):
  # Init Directory
  BASE_DIR = base_dir
  LABEL = label_dir
  DIRECTORY = name

  # Init Saving Path
  PATH = os.path.join(BASE_DIR, DIRECTORY)
  os.mkdir(PATH)
  print(">>> Creating '{}' in '{}'....".format(DIRECTORY, BASE_DIR))

  # Init saving path
  SAVE_IMG_PATH = os.path.join(PATH, "img")
  os.mkdir(SAVE_IMG_PATH)

  # Start image name
  image_name = start_index_img

  # Write annotation to file 
  with open(os.path.join(PATH, annot_name + ".txt"), "a") as f:
    for file in tqdm(os.listdir(img_dir)):
      # read Image
      img = cv.imread(os.path.join(img_dir, file))
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

      # path to annot file
      annot_file = file[2:]
      annot_file = "gt_" + str(int(annot_file.split(".")[0])) + ".txt"

      # read annot file
      with open(os.path.join(label_dir, annot_file),"r") as ff: 
        for line in ff.readlines():
          offsets = line.split(",")
          label = offsets.pop()

          # get ROI
          roi = extract_roi(img, offsets)
          
          # remove invalid label
          if roi.size == 0:
            continue
          elif "#" in label:
            continue
          elif label == " ":
            continue
          elif label == "":
            continue
          else:
            if "\n" not in label:
              label = label + "\n"

            # Save ROI  	
            cv.imwrite(os.path.join(SAVE_IMG_PATH, str(image_name) + ".jpg"), roi)

            # write annotation
            annot = "img/" + str(image_name) + ".jpg" + "\t" + label
            f.write(annot)

            # Annoucement
            #print("Image: {} === Label: {} === Save in {}".format(file, label, str(image_name) + ".jpg"))

            # Increase image_name
            image_name += 1       

# Usage

if __name__ == '__main__':

	## Duong dan den Annotation
	LABEL = ""

	## Duong dan den Image folder
	IMG_DIR = ""

	## Duong dan den thu muc luu cac folder vua tao thanh
	BASE_DIR = ""

	## Run
	generate_data_to_vietocr(BASE_DIR,IMG_DIR,LABEL,1,
	 name = "train_vintext_vietocr_format",
	 annot_name = "train_annotation")
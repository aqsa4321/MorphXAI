import os
import glob

# Paths
config = "configs/rtdetrv3/rtdetrv3_r18vd_6x_trypanosoma.yml"
weights = "output/best_model/model.pdparams"
images_dir = "/home/aqsa/Final_Dataset/images/val/"

# Get all image files
image_files = glob.glob(f"{images_dir}/*.png") + glob.glob(f"{images_dir}/*.jpg") + glob.glob(f"{images_dir}/*.jpeg")

# Run inference for each image
for img in image_files:
    cmd = f'python tools/infer.py -c {config} -o weights={weights} --infer_img={img}'
    print(f"Processing: {os.path.basename(img)}")
    os.system(cmd)
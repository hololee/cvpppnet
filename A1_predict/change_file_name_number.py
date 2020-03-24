import glob
import os
import re

filename = glob.glob('*.png')
for i in filename:
    leng = re.findall(r"(\d+)", i)[0]
    leng = "0" * (3 - len(leng)) + leng
    os.rename(i, "output_" + leng + ".png")

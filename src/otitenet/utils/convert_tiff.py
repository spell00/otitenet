import os, sys
from PIL import Image

def convert(args):
    for fold in os.listdir(f"data/{args.dataset}/Banque_Comert_Turquie_2020"):
        os.makedirs(f"data/{args.dataset}/Banque_Comert_Turquie_2020_jpg/{fold}", exist_ok=True)
        if fold != '.DS_Store':
            for infile in os.listdir(f"data/{args.dataset}/Banque_Comert_Turquie_2020/{fold}"):
                # infile = infile.split(".")[0]
                if infile.split(".")[1] == "tiff":
                    infile = infile.split(".")[0]
                    # print "is tif or bmp"
                    outfile = f"data/{args.dataset}/Banque_Comert_Turquie_2020_jpg/{fold}/{infile}.jpg"
                    infile = f"data/{args.dataset}/Banque_Comert_Turquie_2020/{fold}/{infile}.tiff"
                    im = Image.open(infile)
                    print("new filename : " + outfile)
                    out = im.convert("RGB")
                    out.save(outfile, "JPEG", quality=90)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="QC-CM")
    args = parser.parse_args()
    convert(args)

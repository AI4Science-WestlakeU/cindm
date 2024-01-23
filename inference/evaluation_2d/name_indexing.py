import os
import shutil

def index_name(root):
    full_filelist = os.listdir(root + "boundaries")
    full_filelist.sort()
    filelist = list(set([file[:-5] for file in full_filelist if not file.startswith(".")]))
    filelist.sort()

    print("filelist: ", len(filelist))
    output_folder = os.path.join(root, "renamed_boundaries")
    index_dict = {}
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for i, file in enumerate(filelist):    
        index_dict[file] = i
    print(len(full_filelist))
    for file in full_filelist:
        source_file = os.path.join(root, "boundaries", file)
        index = index_dict[file[:-5]]
        new_filename = "sim_{}_boundary_".format(index) + file[-5:]
        destination_file = os.path.join(output_folder, new_filename)
        shutil.copyfile(source_file, destination_file)

        with open(os.path.join(root, "index_map.txt"), "a") as f:
            f.write(file + ", " + "sim_{}".format(index) + "\n")


if __name__ == "__main__":
    # root = "./double/" # the folder of two cell in Table 3
    root = "./single/" # the folder of one cell in Table 3

    index_name(root)

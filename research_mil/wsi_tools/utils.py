


def get_files(data_txt, check=None):
    # get the contents of a file. i.e. train.txt to list

    try:
        files = []
        with open(data_txt, "r") as f:
            for item in f:
                if check:
                    if check in item:
                        files.append(item.split("\n")[0])
                else:
                    files.append(item.split("\n")[0])
        wsi_files = sorted(files)

        return wsi_files

    except IOError as e:
        print("ERROR loading File {}".format(data_txt), e)
        return
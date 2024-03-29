import os, torch, argparse, yaml




def main(args):

    splits  = ['train', 'val', 'test']

    pth_dir = args['create_lib']['pth_path']
    class_names = args['create_lib']['class_names'].split(',')

    for split in splits:
        root_dir = os.path.join(pth_dir,split)
        
        if os.path.isdir(root_dir):
            print(f'Working on {split}')
            lib = dict()
            for cls in class_names:
                lib_file = torch.load(os.path.join(root_dir,f'{cls}.pth'))
                lib[cls] = lib_file
            
            sv_path = os.path.join(pth_dir,f'{split}_lib.pth')
            torch.save(lib, sv_path)
            print(f'Saved {split} {class_names} to {sv_path}')

    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create library files from the saved pth class files.")
                                                 
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="research_mil/configs/amc_2020/wsi_tools_config_amc.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    main(cfg)
import sys
import yaml
import os
import argparse
import logging
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from wsi_tools.annotation import Formatter  # noqa
from wsi_tools.utils import get_files


def run(args):

    if args['xml2json']['xml_path_dir']: # using folder containing files
        print("using dir")
        xml_dir  = glob.glob(os.path.join(args['xml2json']['xml_path_dir'], '*.xml'))
    elif args['xml2json']['xml_path_file']:
        # use text
        print("using text")
        xml_dir = get_files(args['xml2json']['xml_path_file'], ".xml")
    else:
        logging.INFO('SELECT AN OPTION')
        return

    json_dir = args['xml2json']['json_path_dir']
    # create directory if inexistant
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    print("Found ", len(xml_dir), " files")
    #root_dir = "/media/philipchicco/2.2/Anonymized_Img/"

    for xml_file in xml_dir:
        wsi_name  = (os.path.split(xml_file)[-1]).split(".")[0]
        json_file = os.path.join(json_dir, wsi_name + ".json")
        if not os.path.isfile(xml_file): continue
        print(wsi_name, json_file)
        #xml_file = os.path.join(root_dir, xml_file)
        Formatter.camelyon16xml2json(xml_file, json_file)
    print("done!")


def main(cfg):
    logging.basicConfig(level=logging.INFO)
    run(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert xml format to'
                                                 'internal json format')
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
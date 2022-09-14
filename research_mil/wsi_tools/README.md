## WSI Preprocessing

#### AMC MSS MSI [configs/amc_2020 | wsi_tools ]

1) Ensure you have the WSI split files (train|val|test) consisting the full paths of WSI file and xml annotations.
   e.g. AMC_TRAIN_MSS.txt, AMC_TRAIN_MSS.txt ...
   Place the splits in research_mil/data/splits (see folder for examples)

   **Note**: Modify the paths to point to your files on the server

   : replace ''/media/philipchicco/2.2/Anonymized_Img/" with the actual path on the server.
   : e.g. /path/to/sever/DATA

   ```
   AMC_TRAIN_MSI.txt/
   -- /path/to/WSI_ID1.(tif,svs)
   -- /path/to/WSI_ID1_annot.(xml)
   ```

   1.1) Edit config file in configs folder (see. wsi_tools_config_amc.yml) for each pre-processing step with the desired save paths.
   1.2) Extract annotation points | tissue mask | disease regions | sample patch locations | create library and save patches.

   **Note**: See config file with comments and repeat each step (xml2json,tissue_,tumor,sample) for each class i.e., MSS, MSI text files. Run the following commands:

   ```
   python research_mil/wsi_tools/xml2json.py --config /path/to/amc_2020/config.yml
   python research_mil/wsi_tools/tissue_mask.py --config /path/to/amc_2020/config.yml
   python research_mil/wsi_tools/tumor_mask.py --config /path/to/amc_2020/config.yml
   python research_mil/wsi_tools/sample_spot_gen.py --config /path/to/amc_2020/config.yml
   #################
   python research_mil/wsi_tools/create_lib.py --config /path/to/amc_2020/config.yml
   python research_mil/wsi_tools/patch_gen.py --config /path/to/amc_2020/config.yml
   ```

   (1.3) **tissue_mask.py** : Extracts tissue region from the WSI thumbnail. The default magnification level for all AMC WSI is x40 and tissue extraction is performed at level=6, if you are using WSI from TCGA or other institutes with lower mag (i.e. x20 or x10), the level should be adjusted.

   **Note:** For some WSI that contain 'ink', there is an option to detect and remove ink, check the script and uncomment the related section. Each slide with 'ink' must be verified carefully to ensure the actual tissue region is extracted correctly.

   (1.4) **tumor_mask.py** : This script creates a tumor mask from the reference xml files.

   (1.5) **sample_spot_gen.py** : This script extracts all valid patch co-ordinates for each slide and saves a library file. e.g., mss.pth or msi.pth (see. sample_spot_gen.py for details).

   (1.6) **create_lib.py** : This script will collect the saved 'pth.' library files and combine them into a single library including all classes: For example,

   ```python
   import torch, os   
   # path to where libs are saved for test (i.e. test_mss.pth
   # and test_msi.pth)
   amc_root  = '/path/to/AMC_ALL_LIBS/test'

   # amc test creation
   # load mss and msi libs
   mss_file = torch.load(os.path.join(amc_root,'mss.pth'))
   msi_file = torch.load(os.path.join(amc_root,'msi.pth'))
   torch.save({'mss': mss_file, 'msi': msi_file}, 
   os.path.join(amc_root,'test_lib.pth'))
   print()
   ```

   (1.7) **patch_gen.py** : This script will generate patches from all WSI and saves them in each folder as follows: class | slide_ID | patches

   ```
   /path/to/WSI/patches_m2_l0/
   -- test/
   ---- /0/ # Corresponds to MSS Slides
   ------/WSI_ID_0/ patch_0.png,patch_1.png.... patch_N.png
   ------/WSI_ID_1/ patch_0.png,patch_1.png.... patch_N.png
   ------ 
   ---- /1/ # Corresponds to MSI Slides
   ------ /WSI_ID_0/ patch_0.png,patch_1.png.... patch_N.png
   ------ /WSI_ID_1/ patch_0.png,patch_1.png.... patch_N.png

   ```

   For **MSS/MSI detection** patches are extracted at the highest magnification i.e. (level=0) with each patch having dimension 256x256 and a multiplier of 2 i.e., the multiplier option (m=2) will first extract a patch with size ((m*256) x (mx256)), then resize to 256x256. In addition, all patches are normalized and we ensure that each patch has atleast (>40% tissue). See loader.py and utils_augmentation.py for more details in /wsi_tools/
2) Once pre-processing and patch extraction are complete, copy the library file (e.g. test_lib.pth) to /research_mil/data/amc_2020/msi_test_temp/. For patch training and WSI inference.

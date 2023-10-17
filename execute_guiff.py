import guiff_core.core as core
import os

OUTPUT_DIR = './data/label_targets/'

def get_file_num():
    files = os.listdir(OUTPUT_DIR)
    count = 0
    for file in files:
        if 'label' in file:
            count += 1
    
    return count

SOURCE_FILE = "./data/cctv_raws/20221005/002_20221005_090000_100000.avi"

FILE_NUM = get_file_num()
RECORD_CCTV_NUM = "002"
RECORD_DATE = "221005"
RECORD_HOUR = "09"
START_TIME = "26:48.0"
END_TIME = "27:07.0"

OUTPUT_FILENAME = (
    f"label_{FILE_NUM}_{RECORD_CCTV_NUM}_{RECORD_DATE}_"
    + f"{RECORD_HOUR}{START_TIME[:5].replace(':', '')}_"
    + f"{RECORD_HOUR}{END_TIME[:5].replace(':', '')}.avi"
)

core.trim_video(SOURCE_FILE, START_TIME, END_TIME, OUTPUT_DIR, OUTPUT_FILENAME)

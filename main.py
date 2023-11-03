import time
from datetime import datetime

import video_processor

# Debug mode for displaying image
DEBUG_MODE = False

if __name__ == '__main__':
    print(f'{datetime.now().strftime("%H:%M:%S")}\tProgram has started to process images')
    start_timestamp = time.time()

    video_processor.process()

    end_timestamp = time.time()
    print(f'{datetime.now().strftime("%H:%M:%S")}\tProgram has ended processing')
    print(f'Elapsed time: {end_timestamp - start_timestamp:.2f} seconds')
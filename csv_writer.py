import csv
from datetime import datetime

from common import *

FILENAME = f'processed_data_{TAKE_FRAME_EACH_NTH_SECONDS}.csv'

# Initialize csv writer
# Set headers
HEADERS = [i + ' count' for i in list(CLASSES.values())]
HEADERS.insert(0, 'video timestamp')
HEADERS.append('total vehicles (no trains)')

# Open file
csv_file = open(FILENAME, 'w+', encoding='UTF8', newline='')

# Create writer
csv_writer = csv.writer(csv_file)

# Write headers row
csv_writer.writerow(HEADERS)

def write(timestamp: datetime, data: list):
    # Data output
    row_to_write = data.copy()
    row_to_write.insert(0, timestamp.strftime("%H:%M:%S"))
    row_to_write.append(sum(data) - data[3])

    csv_writer.writerow(row_to_write)

def close():
    csv_file.close()
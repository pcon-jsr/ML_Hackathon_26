# converts integer in every row to 16 bits (2 bytes) and encodes all the bytes to base64 string

import csv
import sys
import base64
import struct

def encode_to_base64(f):
    integers = []

    with open(f, 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            integer_value = int(row[0].strip())
            integers.append(integer_value)

    byte_data = b''.join(struct.pack('>H', num) for num in integers)
    base64_encoded = base64.b64encode(byte_data).decode('ascii')

    print(f"Number of integers:        {len(integers):,}")

    return base64_encoded, byte_data


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        base64_string, byte_data = encode_to_base64(csv_file)

        # stats
        print("\n" + "=" * 70)
        print("ENCODING RESULTS:")
        print("=" * 70)
        print(f"Original size (as bytes):  {len(byte_data):,} bytes")
        print(f"Base64 encoded size:       {len(base64_string):,} characters")
        print("=" * 70)

        with open('integers_base64.txt', 'w') as f:
            f.write(base64_string)
        print("\nOutput saved to integers_base64.txt"); 

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

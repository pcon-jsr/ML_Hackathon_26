# converts every 16 rows of 0/1 to hex

import csv
import sys

def bits_to_int(f):
    integers = []
    bits = []
    
    with open(f, 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            bit = int(row[0].strip())
            bits.append(bit)
            
            if len(bits) == 16:
                integer_val = 0
                for i, bit_val in enumerate(bits):
                    integer_val += bit_val * (2 ** (15 - i))

                hex_val = f"0x{integer_val:04X}"
                
                integers.append(hex_val)
                bits = []
    
    
    return integers


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        integers = bits_to_int(csv_file)
        
        print(f"Total integers generated: {len(integers)}")
        
        # Save to output file
        with open('output-hex.txt', 'w') as f:
            for value in integers:
                f.write(f"{value}\n")
        print("\nOutput saved to 'output-hex.txt'")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

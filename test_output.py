"""
EECS 445 - Fall 2024
Project 1 - test_output.py

Use this script to confirm that your challenge predictions (stored in uniqname.csv) are correctly formatted.
To run this file, run python test_output.py -i uniqname.csv, replacing the file uniqname.csv with your generated output file.

Do not edit test_output.py
"""

import sys
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input data file. Must be in csv.")


def main() -> None:
    """
    Checking the validity of challenge submission.
    """
    args = parser.parse_args()
    input_file = args.input

    if input_file[-4:] != ".csv":
        print("INVALID FILE: file must be of .csv format.")
        sys.exit()

    with open(input_file, newline="") as csvfile:
        filereader = csv.reader(csvfile)
        i = 0
        for row in filereader:
            if i == 0:
                if row[0] != "label":
                    print("INVALID COLUMN NAME: column name is not label.")
                    sys.exit()
            else:
                rating = int(row[0])
                if rating != -1 and rating != 0 and rating != 1:
                    print("INVALID VALUE: values need to be -1, 0, or 1.")
                    sys.exit()
            i += 1
        if i != 2001:
            print("INVALID NUMBER OF ROWS: number of rows is not 2001.")
            sys.exit()
        print("SUCCESS: csv file is valid.")
    return


if __name__ == "__main__":
    main()

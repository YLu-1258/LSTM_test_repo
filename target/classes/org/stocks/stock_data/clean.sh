#!/bin/bash

# Input CSV file
input_file="${1}.csv"

# Output CSV file
output_file="${1}_cleaned.csv"

# Copy input file to output file
cp "$input_file" "$output_file"

# Convert date to Unix timestamp in the output file
awk -F',' 'BEGIN {OFS=","} NR==1 {print; next} { 
    cmd="date -d \""$1"\" +%s" 
    cmd | getline unixtime
    close(cmd)
    $1=unixtime
    print 
}' "$output_file" > tmpfile && mv tmpfile "$output_file"

echo "Conversion completed. Output saved in $output_file"

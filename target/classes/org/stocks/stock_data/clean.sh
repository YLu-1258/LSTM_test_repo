#!/bin/bash

tickers="AMZN COST GOOGL LMT META MSFT NOC TSLA UNH WMT"
for ticker in $tickers; do
    # Input CSV file
    input_file="${ticker}.csv"

    # Output CSV file
    output_file="${ticker}_cleaned.csv"

    # Copy input file to output file
    cp "$input_file" "$output_file"

    # Convert date to Unix timestamp in the output file
    cleaned_data=$(awk -F',' 'BEGIN {OFS=","} NR==1 {print; next} { 
        cmd="date -d \""$1"\" +%s" 
        cmd | getline unixtime
        close(cmd)
        $1=unixtime
        print 
    }' "$output_file")
    cut --complement -f 6 -d, $cleaned_data

    echo "Conversion completed. Output saved in $output_file"

done
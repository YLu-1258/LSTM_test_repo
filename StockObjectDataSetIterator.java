package org.stocks.objects;

import java.io.File;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@NoArgsConstructor
@AllArgsConstructor
@ToString
@Data
public class StockObjectDataSetIterator {

    private String file;
    private double splitRatio;
    private int batchSize;
    private int labelIndex;
    private DataSetIterator iterator;

    public StockObjectDataSetIterator(String directory, String ticker, double splitRatio, int batchSize) {
        this.file = directory + "/" + "ticker" + ".csv";
        this.splitRatio = splitRatio;
        this.batchSize = batchSize;
        this.labelIndex = 4;
        this.iterator = this.extractDataFromCSV();
    }

    private DataSetIterator extractDataFromCSV() {
        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new File(file)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, this.batchSize, this.labelIndex, 1);
        return iterator;
    }
} 

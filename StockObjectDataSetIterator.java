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

    private static String file;
    private double splitRatio;
    private int batchSize;
    private int labelIndex;
    private DataSetIterator iterator;

    public StockObjectDataSetIterator(String directory, String ticker, double splitRatio, int batchSize) {
        file = directory + "/" + "ticker" + "_test.csv";
        this.splitRatio = splitRatio;
        this.batchSize = batchSize;
        this.labelIndex = 4;
        this.iterator = this.extractDataFromCSV();
    }

    public static void main() {
        int features = 4;
        int labels = 1;
        int batchSize = 32;
        int stepCount = 1;
        // TODO: Create CSV file reader, read data into the 3D array, Create INDARRAYS, Create Dataset from each feature and levels, Then create datasetiterator   
        BufferedReader br = new BufferedReader(new FileReader(file));    
        line = br.readLine();
        System.out.println(line);   
        for (int i = 0; i < 140; i ++) {
            double[][][] featureMatrix = new double[stepCount][features][batchSize];
            double[][][] labelsMatrix = new double[stepCount][labels][batchSize];
            for (int batch = 0; batch < batchSize; batch++) {
                featureMatrix[0][0][batch] = 0;// Get CSV data
                featureMatrix[0][1][batch] = 0;//Get CSV data
                featureMatrix[0][2][batch] = 0;// GeT CSV ata
                featureMatrix[0][3][batch] = 0; 
                labelsMatrix[0][0][batch] = 0;
            }
        }
    }
} 

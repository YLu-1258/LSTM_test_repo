package org.stocks.objects;

import java.io.BufferedReader;
import java.io.FileReader;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
        file = directory + "/" + ticker + "_test.csv";
        this.splitRatio = splitRatio;
        this.batchSize = batchSize;
        this.labelIndex = 4;
        this.iterator = this.extractDataFromCSV();
    }

    public static void main() {
        int features = 5;
        int labels = 1;
        int batchSize = 32;
        int stepCount = 1;
        // TODO: Create CSV file reader, read data into the 3D array, Create INDARRAYS, Create Dataset from each feature and levels, Then create datasetiterator   
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            boolean firstLine = true;
            if (firstLine) {
                firstLine = false;
                line = br.readLine();
                continue;
            }
            for (int i = 0; i < 140; i ++) {
                double[][][] featureMatrix = new double[batchSize][stepCount][features];
                double[][][] labelsMatrix = new double[batchSize][stepCount][labels];
                for (int batch = 0; batch < batchSize; batch++) {
                    if ((line = br.readline()) != null) {
                        String[] values = line.split(",");
                        featureMatrix[batch][0][0] = Double.parseDouble(values[0]); // DATE
                        featureMatrix[batch][0][1] = Double.parseDouble(values[1]); // OPEN
                        featureMatrix[batch][0][2] = Double.parseDouble(values[2]); // HIGH
                        featureMatrix[batch][0][3] = Double.parseDouble(values[4]); // LOW
                        featureMatrix[batch][0][4] = Double.parseDouble(values[5]); // VOLUME
                        labelsMatrix[batch][0][0] = Double.parseDouble(values[3]); // CLOSE
                    }
                    INDArray featuresArray = Nd4j.create(featureMatrix);
                    INDArray labelsArray = Nd4j.create(labelsMatrix);
                    Dataset dataset = new DataSet(featuresArray, labelsArray);
                    // create INDArray
                    // create DataSet
                }
                // Add to DatasetIterator
            }
            
        }    
        
    }
} 

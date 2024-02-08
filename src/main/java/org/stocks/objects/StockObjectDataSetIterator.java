package org.stocks.objects;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.File;
import java.util.Arrays;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import lombok.val;
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
    private int labelIndex;
    private int features; //5
    private int labels; // 1
    private int batchSize; //32
    private int stepCount; //1
    private ROC roc;
    private DataSetIterator iterator;

    public StockObjectDataSetIterator(String directory, String ticker, int features, int labels, int batchSize, int stepCount) {
        this.file = directory + "/" + ticker + "_test.csv";
        this.features = features;
        this.labels = labels;
        this.batchSize = batchSize;
        this.stepCount = stepCount;
        this.roc = new ROC(100);
        // this.splitRatio = splitRatio;
        // this.batchSize = batchSize;
        // this.labelIndex = 4;
    }

    public void TrainModel(MultiLayerNetwork net) {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            line = br.readLine();
            for (int i = 0; i < 120; i ++) {
                double[][][] featureMatrix = new double[batchSize][this.features][this.stepCount];
                double[][][] labelsMatrix = new double[batchSize][this.labels][this.stepCount];
                for (int batch = 0; batch < this.batchSize; batch++) {
                    line = br.readLine();
                    String[] values = line.split(",");
                    featureMatrix[batch][0][0] = Double.parseDouble(values[0]); // DATE
                    featureMatrix[batch][1][0] = Double.parseDouble(values[1]); // OPEN
                    featureMatrix[batch][2][0] = Double.parseDouble(values[2]); // HIGH
                    featureMatrix[batch][3][0] = Double.parseDouble(values[4]); // LOW
                    featureMatrix[batch][4][0] = Double.parseDouble(values[5]); // VOLUME
                    labelsMatrix[batch][0][0] = Double.parseDouble(values[3]); // CLOSE
                }
                INDArray featuresArray = Nd4j.create(featureMatrix);
                INDArray labelsArray = Nd4j.create(labelsMatrix);
                // System.out.println(labelsArray);
                DataSet train = new DataSet(featuresArray, labelsArray);
                // System.out.println("Training");
                net.fit(train);
                net.rnnClearPreviousState();
                // file locationToSave = new File("src/main/resources/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
                // ModelSerializer.writeModel(net, locationToSave, true);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void TestModel(MultiLayerNetwork net) {
        
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = br.readLine();
            for (int i = 0; i < 120*32; i++) {
                line = br.readLine();
            }
            for (int i = 0; i < 20; i++) {
                double[][][] featureMatrix = new double[batchSize][this.features][stepCount];
                double[][][] labelsMatrix = new double[batchSize][this.labels][stepCount];
                for (int batch = 0; batch < this.batchSize; batch++) {
                    line = br.readLine();
                    String[] values = line.split(",");
                    featureMatrix[batch][0][0] = Double.parseDouble(values[0]); // DATE
                    featureMatrix[batch][1][0] = Double.parseDouble(values[1]); // OPEN
                    featureMatrix[batch][2][0] = Double.parseDouble(values[2]); // HIGH
                    featureMatrix[batch][3][0] = Double.parseDouble(values[4]); // LOW
                    featureMatrix[batch][4][0] = Double.parseDouble(values[5]); // VOLUME
                    labelsMatrix[batch][0][0] = Double.parseDouble(values[3]); // CLOSE
                }
                INDArray featuresArray = Nd4j.create(featureMatrix);
                INDArray labelsArray = Nd4j.create(labelsMatrix);
                // System.out.println(Arrays.deepToString(featureMatrix));
                // System.out.println(Arrays.deepToString(labelsMatrix)+"\n\n\n\n\n\n");
                // System.out.println("Features:" + featuresArray);
                // System.out.println("Labels:" + labelsArray);
                DataSet test = new DataSet(featuresArray, labelsArray);
                INDArray output = net.output(test.getFeatures(), false);
                System.out.println("Output: ");
                System.out.println(output);
                System.out.println("Actual: ");
                // System.out.println(test.getFeatures());
                System.out.println(test.getLabels());
                roc.evalTimeSeries(test.getLabels(), output);
            }
            System.out.println("FINAL TEST AUC: " + roc.calculateAUC());
            File locationToSave = new File("src/main/resources/StockPriceLSTM_".concat("CLOSE").concat(".zip"));
            ModelSerializer.writeModel(net, locationToSave, true);
            net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    public void TrainAndTestModel(MultiLayerNetwork net) {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            line = br.readLine();
            for (int i = 0; i < 120; i ++) {
                double[][][] featureMatrix = new double[batchSize][this.features][this.stepCount];
                double[][][] labelsMatrix = new double[batchSize][this.labels][this.stepCount];
                for (int batch = 0; batch < this.batchSize; batch++) {
                    line = br.readLine();
                    String[] values = line.split(",");
                    featureMatrix[batch][0][0] = Double.parseDouble(values[0]); // DATE
                    featureMatrix[batch][1][0] = Double.parseDouble(values[1]); // OPEN
                    featureMatrix[batch][2][0] = Double.parseDouble(values[2]); // HIGH
                    featureMatrix[batch][3][0] = Double.parseDouble(values[4]); // LOW
                    featureMatrix[batch][4][0] = Double.parseDouble(values[5]); // VOLUME
                    labelsMatrix[batch][0][0] = Double.parseDouble(values[3]); // CLOSE
                }
                INDArray featuresArray = Nd4j.create(featureMatrix);
                INDArray labelsArray = Nd4j.create(labelsMatrix);
                // System.out.println(labelsArray);
                DataSet train = new DataSet(featuresArray, labelsArray);
                // System.out.println("Training");
                net.fit(train);
                net.rnnClearPreviousState();
                // file locationToSave = new File("src/main/resources/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
                // ModelSerializer.writeModel(net, locationToSave, true);
                DataSet test = new DataSet(featuresArray, labelsArray);
                INDArray output = net.output(test.getFeatures(), false);
                System.out.println("Output: ");
                System.out.println(output);
                System.out.println("Actual: ");
                // System.out.println(test.getFeatures());
                System.out.println(test.getLabels());
                roc.evalTimeSeries(test.getLabels(), output);
            }
            System.out.println("FINAL TEST AUC: " + roc.calculateAUC());
            File locationToSave = new File("src/main/resources/StockPriceLSTM_".concat("CLOSE").concat(".zip"));
            ModelSerializer.writeModel(net, locationToSave, true);
            net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

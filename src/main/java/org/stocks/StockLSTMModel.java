package org.stocks;

import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;

import org.stocks.model.LSTMNetModel;

public class StockLSTMModel {

    //private static final Logger log = LoggerFactory.getLogger(StockLSTMModel.class);

    private static int exampleLength = 22; // time series length, assume 22 working days per month

    public static void main (String[] args) throws IOException {
        String directory = "/home/eris29/APCSA/LSTM_test_repo/src/main/java/org/stocks/stock_data";
        String ticker = "AAPL"; // stock name
        String file = directory + "/" + ticker + "_cleaned.csv";
        double splitRatio = 0.9; // 90% for training, 10% for testing
        int epochs = 100; // training epochs

        // Number of input features
        int labelIndex = 4; // Update according to your CSV file

        // Number of classes (if applicable)
        int numClasses = 3; // Update according to your CSV file
 
        // Batch size
        int batchSize = 32; // Set according to your needs

        System.out.println("Create dataSet iterator...");
        int numLinesToSkip = 2;
        char delimiter = ',';
        try (RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter)) {
            recordReader.initialize(new FileSplit(new File(file)));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, labelIndex, true);

            MultiLayerNetwork net = LSTMNetModel.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
            DataSet set = iterator.next();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
       
        // Start training
        


        // Define iterator
        
        
    }
        // System.out.println("Load test dataset...");
        // List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

    //     System.out.println("Build lstm networks...");
    //     MultiLayerNetwork net = LSTMNetModel.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());

    //     // System.out.println("Training...");
    //     // System.out.println(iterator.getTrain());
    //     // for (int i = 0; i < epochs; i++) {
    //     //     while (iterator.hasNext()) {
    //     //         net.fit(iterator.next());
    //     //     } // fit model using mini-batch data
    //     //     iterator.reset(); // reset iterator
    //     //     net.rnnClearPreviousState(); // clear previous state
    //     // }

    //     System.out.println("Saving model...");
    //     File locationToSave = new File("src/main/resources/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
    //     // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
    //     ModelSerializer.writeModel(net, locationToSave, true);

    //     System.out.println("Load model...");
    //     net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

    //     System.out.println("Testing...");
    //     if (category.equals(PriceCategory.ALL)) {
    //         INDArray max = Nd4j.create(iterator.getMaxArray());
    //         INDArray min = Nd4j.create(iterator.getMinArray());
    //         predictAllCategories(net, test, max, min);
    //     } else {
    //         double max = iterator.getMaxNum(category);
    //         double min = iterator.getMinNum(category);
    //         predictPriceOneAhead(net, test, max, min, category);
    //     }
    //     System.out.println("Done...");
    // }

    // /** Predict one feature of a stock one-day ahead */
    // private static void predictPriceOneAhead (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
    //     double[] predicts = new double[testData.size()];
    //     double[] actuals = new double[testData.size()];
    //     for (int i = 0; i < testData.size(); i++) {
    //         predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
    //         actuals[i] = testData.get(i).getValue().getDouble(0);
    //     }
    //     System.out.println("Print out Predictions and Actual Values...");
    //     System.out.println("Predict,Actual");
    //     for (int i = 0; i < predicts.length; i++) System.out.println(predicts[i] + "," + actuals[i]);
    //     System.out.println("Plot...");
    //     StockGraph.plot(predicts, actuals, String.valueOf(category));
    // }

    // /** Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead */
    // private static void predictAllCategories (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
    //     INDArray[] predicts = new INDArray[testData.size()];
    //     INDArray[] actuals = new INDArray[testData.size()];
    //     for (int i = 0; i < testData.size(); i++) {
    //         predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
    //         actuals[i] = testData.get(i).getValue();
    //     }
    //     System.out.println("Print out Predictions and Actual Values...");
    //     System.out.println("Predict\tActual");
    //     for (int i = 0; i < predicts.length; i++) System.out.println(predicts[i] + "\t" + actuals[i]);
    //     System.out.println("Plot...");
    //     for (int n = 0; n < 5; n++) {
    //         double[] pred = new double[predicts.length];
    //         double[] actu = new double[actuals.length];
    //         for (int i = 0; i < predicts.length; i++) {
    //             pred[i] = predicts[i].getDouble(n);
    //             actu[i] = actuals[i].getDouble(n);
    //         }
    //         String name;
    //         switch (n) {
    //             case 0: name = "Stock OPEN Price"; break;
    //             case 1: name = "Stock CLOSE Price"; break;
    //             case 2: name = "Stock LOW Price"; break;
    //             case 3: name = "Stock HIGH Price"; break;
    //             case 4: name = "Stock VOLUME Amount"; break;
    //             default: throw new NoSuchElementException();
    //         }
    //         StockGraph.plot(pred, actu, name);
    //     }
    // }

}
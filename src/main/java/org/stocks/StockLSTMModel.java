package org.stocks;
import java.util.ArrayList;
import java.io.BufferedReader;  
import java.io.IOException;
import java.io.FileReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

public class StockLSTMModel {
    private static String ticker = "AAPL";
    private static String tickerPath = "/home/eris29/APCSA/deeplearning4j-examples/nd4j-ndarray-examples/src/main/java/org/nd4j/examples/quickstart/stock_data/" + ticker + ".csv";;
    

    private static double[] extractCSVCloseData(){
        ArrayList<Double> CSVData = new ArrayList<Double>();
        try (BufferedReader br = new BufferedReader(new FileReader(tickerPath))) {
            int closeColumnIndex = 4;
            String line = br.readLine();  
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                CSVData.add(Double.parseDouble(data[closeColumnIndex]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] CSVDataArray = new double[CSVData.size()];
        for (int i = 0; i < CSVData.size(); i++) {
            CSVDataArray[i] = CSVData.get(i);
        }

        return CSVDataArray;
    }

    private static INDArray createScaledData() {
        double[] CSVDataArray = extractCSVCloseData();
        INDArray data = Nd4j.create(CSVDataArray);
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        return data;
    }

    public static void main(String args[]) {
        INDArray scaledData = createScaledData();
        System.out.println(scaledData);
    }
}
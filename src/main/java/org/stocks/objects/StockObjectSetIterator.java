package org.stocks.objects;

import com.google.common.collect.ImmutableMap;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;


public class StockObjectSetIterator implements DataSetIterator {

    /** category and its index */
    private final Map<PriceCategory, Integer> featureMapIndex = ImmutableMap.<PriceCategory, Integer>builder()
        .put(PriceCategory.OPEN, 0)
        .put(PriceCategory.HIGH, 1)
        .put(PriceCategory.LOW, 2)
        .put(PriceCategory.CLOSE, 3)
        .put(PriceCategory.ADJCLOSE, 4)
        .put(PriceCategory.VOLUME, 5)
        .build();

    private final int VECTOR_SIZE = 6; // number of features for a stock data
    private int miniBatchSize; // mini-batch size
    private int exampleLength = 22; // default 22, say, 22 working days per month
    private int predictLength = 1; // default 1, say, one day ahead prediction

    /** minimal values of each feature in stock dataset */
    private double[] minArray = new double[VECTOR_SIZE];
    /** maximal values of each feature in stock dataset */
    private double[] maxArray = new double[VECTOR_SIZE];

    /** feature to be selected as a training target */
    private PriceCategory category;

    /** mini-batch offset */
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    /** stock dataset for training */
    private List<StockObject> train;
    /** adjusted stock dataset for testing */
    private List<Pair<INDArray, INDArray>> test;

    public StockObjectSetIterator (String filename, String ticker, int miniBatchSize, int exampleLength, double splitRatio, PriceCategory category) {
        List<StockObject> StockObjectList = readStockObjectFromFile(filename, ticker);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(StockObjectList.size() * splitRatio);
        train = StockObjectList.subList(0, split);
        test = generateTestDataSet(StockObjectList.subList(split, StockObjectList.size()));
        initializeOffsets();
    }

    /** initialize the mini-batch offsets */
    private void initializeOffsets () {
        exampleStartOffsets.clear();
        int window = exampleLength + predictLength;
        for (int i = 0; i < train.size() - window; i++) { exampleStartOffsets.add(i); }
    }

    public List<Pair<INDArray, INDArray>> getTestDataSet() { return test; }

    public double[] getMaxArray() { return maxArray; }

    public double[] getMinArray() { return minArray; }

    public double getMaxNum (PriceCategory category) { return maxArray[featureMapIndex.get(category)]; }

    public double getMinNum (PriceCategory category) { return minArray[featureMapIndex.get(category)]; }

    @Override
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label;
        if (category.equals(PriceCategory.ALL)) label = Nd4j.create(new int[] {actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        else label = Nd4j.create(new int[] {actualMiniBatchSize, predictLength, exampleLength}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            StockObject curData = train.get(startIdx);
            StockObject nextData;
            for (int i = startIdx; i < endIdx; i++) {
                int c = i - startIdx;
                input.putScalar(new int[] {index, 0, c}, (curData.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
                input.putScalar(new int[] {index, 1, c}, (curData.getHigh() - minArray[1]) / (maxArray[1] - minArray[1]));
                input.putScalar(new int[] {index, 2, c}, (curData.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
                input.putScalar(new int[] {index, 3, c}, (curData.getClose() - minArray[3]) / (maxArray[3] - minArray[3]));
                input.putScalar(new int[] {index, 4, c}, (curData.getAdjClose() - minArray[4]) / (maxArray[4] - minArray[4]));
                input.putScalar(new int[] {index, 5, c}, (curData.getVolume() - minArray[5]) / (maxArray[5] - minArray[5]));
                nextData = train.get(i + 1);
                if (category.equals(PriceCategory.ALL)) {
                    input.putScalar(new int[] {index, 0, c}, (nextData.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
                    input.putScalar(new int[] {index, 1, c}, (nextData.getHigh() - minArray[1]) / (maxArray[1] - minArray[1]));
                    input.putScalar(new int[] {index, 2, c}, (nextData.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
                    input.putScalar(new int[] {index, 3, c}, (nextData.getClose() - minArray[3]) / (maxArray[3] - minArray[3]));
                    input.putScalar(new int[] {index, 4, c}, (nextData.getAdjClose() - minArray[4]) / (maxArray[4] - minArray[4]));
                    input.putScalar(new int[] {index, 5, c}, (nextData.getVolume() - minArray[5]) / (maxArray[5] - minArray[5]));
                } else {
                    label.putScalar(new int[]{index, 0, c}, feedLabel(nextData));
                }
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    private double feedLabel(StockObject data) {
        double value;
        switch (category) {
            case OPEN: value = (data.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]); break;
            case HIGH: value = (data.getHigh() - minArray[1]) / (maxArray[1] - minArray[1]); break;
            case LOW: value = (data.getLow() - minArray[2]) / (maxArray[2] - minArray[2]); break;
            case CLOSE: value = (data.getClose() - minArray[3]) / (maxArray[3] - minArray[3]); break;
            case ADJCLOSE: value = (data.getAdjClose() - minArray[4]) / (maxArray[4] - minArray[4]); break;
            case VOLUME: value = (data.getVolume() - minArray[5]) / (maxArray[5] - minArray[5]); break;
            default: throw new NoSuchElementException();
        }
        return value;
    }

    // @Override public int totalExamples() { return train.size() - exampleLength - predictLength; }

    @Override public int inputColumns() { return VECTOR_SIZE; }

    @Override public int totalOutcomes() {
        if (this.category.equals(PriceCategory.ALL)) return VECTOR_SIZE;
        else return predictLength;
    }

    @Override public boolean resetSupported() { return false; }

    @Override public boolean asyncSupported() { return false; }

    @Override public void reset() { initializeOffsets(); }

    @Override public int batch() { return miniBatchSize; }

    // @Override public int cursor() { return totalExamples() - exampleStartOffsets.size(); }

    // @Override public int numExamples() { return totalExamples(); }

    @Override public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override public DataSetPreProcessor getPreProcessor() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override public List<String> getLabels() { throw new UnsupportedOperationException("Not Implemented"); }

    @Override public boolean hasNext() { return exampleStartOffsets.size() > 0; }

    @Override public DataSet next() { return next(miniBatchSize); }
    
    private List<Pair<INDArray, INDArray>> generateTestDataSet (List<StockObject> StockObjectList) {
    	int window = exampleLength + predictLength;
    	List<Pair<INDArray, INDArray>> test = new ArrayList<>();
    	for (int i = 0; i < StockObjectList.size() - window; i++) {
    		INDArray input = Nd4j.create(new int[] {exampleLength, VECTOR_SIZE}, 'f');
    		for (int j = i; j < i + exampleLength; j++) {
    			StockObject stock = StockObjectList.get(j);
    			input.putScalar(new int[] {j - i, 0}, (stock.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
    			input.putScalar(new int[] {j - i, 1}, (stock.getHigh() - minArray[1]) / (maxArray[1] - minArray[1]));
    			input.putScalar(new int[] {j - i, 2}, (stock.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
    			input.putScalar(new int[] {j - i, 3}, (stock.getClose() - minArray[3]) / (maxArray[3] - minArray[3]));
    			input.putScalar(new int[] {j - i, 4}, (stock.getAdjClose() - minArray[4]) / (maxArray[4] - minArray[4]));
                input.putScalar(new int[] {j - i, 5}, (stock.getVolume() - minArray[5]) / (maxArray[5] - minArray[5]));
    		}
            StockObject stock = StockObjectList.get(i + exampleLength);
            INDArray label;
            if (category.equals(PriceCategory.ALL)) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f'); // ordering is set as 'f', faster construct
                label.putScalar(new int[] {0}, stock.getOpen());
                label.putScalar(new int[] {1}, stock.getHigh());
                label.putScalar(new int[] {2}, stock.getLow());
                label.putScalar(new int[] {3}, stock.getClose());
                label.putScalar(new int[] {4}, stock.getAdjClose());
                label.putScalar(new int[] {5}, stock.getVolume());
            } else {
                label = Nd4j.create(new int[] {1}, 'f');
                switch (category) {
                    case OPEN: label.putScalar(new int[] {0}, stock.getOpen()); break;
                    case HIGH: label.putScalar(new int[] {0}, stock.getHigh()); break;
                    case LOW: label.putScalar(new int[] {0}, stock.getLow()); break;
                    case CLOSE: label.putScalar(new int[] {0}, stock.getClose()); break;
                    case ADJCLOSE: label.putScalar(new int[] {0}, stock.getAdjClose()); break;
                    case VOLUME: label.putScalar(new int[] {0}, stock.getVolume()); break;
                    default: throw new NoSuchElementException();
                }
            }
    		test.add(new Pair<>(input, label));
    	}
    	return test;
    }

	private List<StockObject> readStockObjectFromFile (String filename, String ticker) {
        List<StockObject> StockObjectList = new ArrayList<>();
        try {
            for (int i = 0; i < maxArray.length; i++) { // initialize max and min arrays
                maxArray[i] = Double.MIN_VALUE;
                minArray[i] = Double.MAX_VALUE;
            }
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll(); // load all elements in a list
            for (String[] arr : list) {
                System.out.println(arr);
                if (!arr[1].equals(ticker)) continue;
                double[] nums = new double[VECTOR_SIZE];
                for (int i = 0; i < arr.length - 2; i++) {
                    nums[i] = Double.valueOf(arr[i + 2]);
                    if (nums[i] > maxArray[i]) maxArray[i] = nums[i];
                    if (nums[i] < minArray[i]) minArray[i] = nums[i];
                }
                StockObjectList.add(new StockObject(arr[0], arr[1], nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return StockObjectList;
    }
}